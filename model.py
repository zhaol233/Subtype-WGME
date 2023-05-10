# -*- coding: utf-8 -*-

import math
import numpy as np
import torch.nn as nn
import torch
import pandas as pd

import torch.nn.functional as F
from timm.models.layers import Mlp
from timm.models.mlp_mixer import MlpMixer
from torch.autograd import Variable
from utils.logger import log
from utils.sur_analysis import gmm, get_p



class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class EarlyStopping:
    """Early stops the training if survival p value get significant."""

    def __init__(self,
                 cancer_type,
                 n_cluster,
                 valid_data,
                 max_stop_epoch,
                 warmup=10,
                 patience=3,
                 min_stop_epoch=20,
                 verbose=False):
        """
        Args:
            warmup: skip epoch
            patience (int): How long to wait after last time validation 
                p value improved. Default: 3
            min_stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each 
                validation pvalue improvement. Default: False
        """
        self.cancer_type = cancer_type
        self.n_cluster = n_cluster
        self.warmup = warmup
        self.sig_p_value = 1.301
        self.patience = patience
        self.min_stop_epoch = min_stop_epoch
        self.max_stop_epoch = max_stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_p_value = 0.
        self.valid_data = valid_data
        self.early_stop = False

        df_clinic_path = f'../results/{cancer_type}/clinic.csv'
        self.df_clinic = pd.read_csv(df_clinic_path, index_col=0)
        self.sample_name = self.df_clinic.index
        log.info(f"---------------early stop model init----------")
        log.info(f"cancer_type={self.cancer_type}, n_cluster={self.n_cluster}, sig_p_value={self.sig_p_value},"
                 f"patience={self.patience}")

        self.feature = pd.DataFrame()
        self.cluster = pd.DataFrame()

    def __call__(self, epoch, model):
        df_fea = self.get_feature(self.valid_data, model)
        df_cluster = self.get_cluster(df_fea)
        p_value = get_p(self.cancer_type, df_c=df_cluster, df_l=self.df_clinic)

        if epoch < self.warmup:
            pass
        elif p_value < self.sig_p_value:
            if self.best_p_value > 0:
                self.counter += 1
                log.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch >= self.min_stop_epoch or epoch == self.max_stop_epoch:
                    self.save_feature(self.feature, self.cluster)
                    self.early_stop = True
        elif p_value >= self.sig_p_value:
            if p_value > self.best_p_value:
                self.best_p_value = p_value
                self.feature = df_fea
                self.cluster = df_cluster
                self.counter = 0
                if epoch == self.max_stop_epoch:
                    self.save_feature(self.feature, self.cluster)
                    self.early_stop = True
            else:
                self.counter += 1
                log.info(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
                if self.counter >= self.patience and epoch >= self.min_stop_epoch:
                    self.save_feature(self.feature, self.cluster)
                    self.early_stop = True
        else:
            log.info(
                f'Cannot EarlyStopping'
            )

    def save_feature(self, df_fea, df_cluster):
        '''Saves model when p value can not get better.'''
        df_fea.to_csv(f"../fea/{self.cancer_type}/{self.cancer_type}.fea", header=True, index=True, sep=',')
        df_cluster.to_csv(f"../results/{self.cancer_type}/{self.cancer_type}.SubtypeWGME", header=True, index=True,
                          sep=',')

    def get_feature(self, valid_data, model):
        with torch.no_grad():
            vec = model.encode(valid_data).cpu()
            vec = vec.detach().numpy()
            vec = pd.DataFrame(vec)
            vec.index = self.sample_name
        return vec

    def get_cluster(self, vec):
        df_cluster = gmm(self.n_cluster,
                         cancer_type=self.cancer_type,
                         fea_df=vec,
                         save=False)
        return df_cluster


class ViTEncoder(nn.Module):
    def __init__(self,
                 data_type,
                 data_size,
                 img_size=224,
                 latent_dim=256,
                 act_layer=None,
                 train=True):
        super(ViTEncoder, self).__init__()
        self.patch_size = 16

        self.latent_dim = latent_dim
        self.use_gpu = torch.cuda.is_available()

        self.train = train
        self.length = sum(data_size)

        self.idx = np.arange(0, self.length)
        np.random.shuffle(self.idx)

        # multi input
        self.weight = torch.Tensor([1 / len(data_size) for dim in data_size])  # sample weight for each omics
        self.dim = [int(i * self.latent_dim) + 1 for i in self.weight]
        if sum(self.dim) != self.latent_dim:
            self.dim[-1] += self.latent_dim - sum(self.dim)

        self.multi_input_encode = nn.ModuleList()
        for i, omics in enumerate(data_type):
            self.multi_input_encode.add_module(
                f"encode_{omics}",
                nn.Sequential(
                    Mlp(data_size[i], latent_dim, self.dim[i], drop=0.2),
                ))

        # MLP-mixer
        self.img_size = img_size
        self.encoder = MlpMixer(in_chans=1,
                                img_size=self.img_size,
                                embed_dim=self.latent_dim,
                                drop_rate=0.4,
                                drop_path_rate=0.4,
                                mlp_ratio=(0.5, 0.5),
                                num_blocks=8)

        # to_latent
        self.res_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.GELU())

    def data_process(self, x):
        x = torch.cat(x, dim=1)
        x = x[:, self.idx]  # shuffle

        batch_size, feature_size = x.shape
        padding_token = torch.zeros(
            abs(self.img_size * self.img_size - feature_size)).repeat(
            batch_size, 1)
        if self.use_gpu:
            padding_token = padding_token.cuda()

        res = torch.cat([x, padding_token], dim=1)
        res = res.reshape(batch_size, 1, self.img_size, self.img_size)
        return res

    def forward(self, x):
        # multiple
        multi_res = []
        for i, dense_block in enumerate(self.multi_input_encode):
            omics_x = dense_block(x[i])
            multi_res.append(omics_x)
        multi_res = torch.cat(multi_res, dim=1)

        # mlp mixer
        x = self.data_process(x)
        x = self.encoder.forward_features(x)
        x = x.mean(dim=1)

        x = x + multi_res
        x = self.res_mlp(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, data_type, data_size, img_size=224, latent_dim=256):
        super(ViTDecoder, self).__init__()
        self.length = sum(data_size)
        self.latent_dim = latent_dim
        self.img_size = img_size

        # multi decoder
        self.multi2x = nn.Linear(latent_dim, latent_dim)
        self.multi_decoder = nn.ModuleList()
        for name, size in zip(data_type, data_size):
            self.multi_decoder.add_module(
                f"decoder_{name}",
                nn.Sequential(
                    Mlp(latent_dim,
                        int(latent_dim * 0.25),
                        latent_dim,
                        act_layer=nn.ReLU),
                    nn.Linear(latent_dim, latent_dim),
                    nn.Linear(latent_dim, size)))

        # img decoder
        self.vit_x = nn.Sequential(
            Mlp(latent_dim,
                int(latent_dim * 0.5),
                latent_dim,
                drop=0.2,
                act_layer=nn.ReLU),
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, self.img_size * self.img_size),
            View((-1, 1, self.img_size, self.img_size)),
        )

    def forward(self, x):
        multi_res = []

        for dense in self.multi_decoder:
            multi_res.append(dense(x))
        img_x = self.vit_x(x)
        return [multi_res, img_x]


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.desc = nn.Sequential(nn.Linear(latent_dim, latent_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(latent_dim * 4, latent_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(latent_dim * 4, latent_dim),
                                  nn.ReLU(),
                                  nn.Linear(latent_dim, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.desc(x)
        return x


class SubtypeWGME(nn.Module):
    def __init__(self, data_type, data_size, latent_dim=256):
        super(SubtypeWGME, self).__init__()
        log.info(f" ------------ SubtypeWGME model init  ------------")

        self.use_gpu = torch.cuda.is_available()

        self.latent_dim = latent_dim
        self.length = sum(data_size)

        self.patch_size = 16
        self.latent_dim = self.patch_size * self.patch_size

        self.img_size = math.ceil(
            math.sqrt(self.length) / self.patch_size) * self.patch_size

        log.info(f"data_type: {data_type}, data_size: {data_size}")
        log.info(f"latent_dim={latent_dim}, patch_size={self.patch_size}, img_size={self.img_size}")

        self.encoder = ViTEncoder(data_type, data_size, self.img_size,
                                  self.latent_dim)
        self.encoder_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.encoder_log_var = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder = ViTDecoder(data_type, data_size, self.img_size,
                                  self.latent_dim)
        self.disc = Discriminator(self.latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return self.reparameterize(mean, log_var)

    def reparameterize(self, mean, log_var):
        """
        mean + randn * e**( log_var * 0.5)
        """
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        if self.use_gpu:
            eps = eps.cuda()
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mean)

    def decode(self, x):
        con_x = self.decoder(x)
        return con_x

    def forward(self, x):
        feature = self.encode(x)
        disc_res = self.disc(feature)
        con_x = self.decode(feature)
        return feature, disc_res, con_x


def loss_kd(outputs, targets, labels, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss


class SHAP_classifier(nn.Module):
    def __init__(self, data_size, latent_dim, cluster_num):
        super(SHAP_classifier, self).__init__()
        log.info("SHAP classifier init -----")
        self.length = sum(data_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.length, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, cluster_num)
        )

    def forward(self, x):
        return self.classifier(x)

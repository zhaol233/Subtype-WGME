# -*- coding: utf-8 -*-
import math
import os
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


class ValidModel:
    """Early stops the training if survival p value get significant."""

    def __init__(self,
                 cancer_type,
                 n_cluster,
                 valid_data,
                 clinic_data
                ):
        self.cancer_type = cancer_type
        self.n_cluster = n_cluster
        self.valid_data = valid_data
        self.df_clinic = clinic_data
        self.sample_name = self.df_clinic.index
        log.info("---------------valid model model init----------")
        log.info(f"cancer_type={self.cancer_type}, n_cluster={self.n_cluster}")

        self.feature = pd.DataFrame()
        self.cluster = pd.DataFrame()

    def __call__(self, epoch, model):
        # pass
        self.feature = self.get_feature(self.valid_data, model)
        df_cluster = self.get_cluster(self.feature)
        if df_cluster is None:
            print("None")
            return
        p_value = get_p(self.cancer_type, df_c=df_cluster, df_l=self.df_clinic)
        print(epoch, p_value)


    def get_feature(self, valid_data, model):
        model.eval()
        with torch.no_grad():
            vec = model.encode(valid_data).cpu()
            vec = vec.detach().numpy()
            vec = pd.DataFrame(vec)
            vec.index = self.sample_name
        model.train()
        return vec

    def get_cluster(self, vec):
        df_cluster = gmm(self.n_cluster,
                         cancer_type=self.cancer_type,
                         fea_df=vec,
                         save=False)
        return df_cluster

    def getp(self, model):
        self.feature = self.get_feature(self.valid_data, model)
        df_cluster = self.get_cluster(self.feature)
        p_value = get_p(self.cancer_type, df_c=df_cluster, df_l=self.df_clinic)
        df_cluster.to_csv(f"../results/{self.cancer_type}/{self.cancer_type}.SubtypeWGME", header=True, index=True,
                          sep=',')
        self.feature.to_csv(f"../fea/{self.cancer_type}/{self.cancer_type}.fea", header=True, index=True, sep=',')
        print(p_value)

    def save(self, model):
        os.makedirs("./model",exist_ok=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
        torch.save(model.state_dict(), f"./model/{self.cancer_type}.pt")
        self.feature = self.get_feature(self.valid_data, model)
        self.feature.to_csv(f"../fea/{self.cancer_type}/{self.cancer_type}.fea", header=True, index=True, sep=',')
        self.cluster = self.get_cluster(self.feature)
        self.cluster.to_csv(f"../results/{self.cancer_type}/{self.cancer_type}.SubtypeWGME", header=True, index=True,
                          sep=',')
        print(get_p(self.cancer_type, df_c=self.cluster, df_l=self.df_clinic))

        
class ViTEncoder(nn.Module):
    def __init__(self,
                 data_type,
                 data_size,
                 img_size=224,
                 latent_dim=256,
                 act_layer=None):
        super(ViTEncoder, self).__init__()
        self.patch_size = 16

        self.latent_dim = latent_dim
        self.use_gpu = torch.cuda.is_available()

        # self.train = train
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
            nn.GELU()
        )

    def data_process(self, x):
        x = torch.cat(x, dim=1)
        # x = x[:, self.idx]  # shuffle

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
                        act_layer=nn.ReLU
                       ),
                    nn.Linear(latent_dim, latent_dim),
                    nn.Linear(latent_dim, size)
                )
            )

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
        self.desc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
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
        log.info(" ------------ SubtypeWGME model init  ------------")

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

        self.eps = nn.Parameter(torch.FloatTensor(self.latent_dim).normal_(), requires_grad=False)
        # print(self.eps)

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return self.reparameterize(mean, log_var)

    def reparameterize(self, mean, log_var):
        """
        mean + randn * e**( log_var * 0.5)
        """
        if not self.training:
            std = log_var.mul(0.5).exp_()
            return self.eps.mul(std).add_(mean)
        else:
            eps = Variable(torch.FloatTensor(self.latent_dim).normal_())
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


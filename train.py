# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 15:38
# @Author  : zhaoliang
# @Description: TODO
import argparse
import time
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from numpy import ravel

from utils.myutils import print_environment_info, provide_determinism, cancer_dict
from utils.logger import log
from model import SubtypeWGME, EarlyStopping
import os

parser = argparse.ArgumentParser(description='SubtypeWGME v1.0')
parser.add_argument("-e", dest='epochs', type=int, default=200, help="number of max iterations")
parser.add_argument("-m", dest='run_mode', default="SubtypeWGME", help="SubtypeWGME, biomarker_rf")

parser.add_argument("-t", dest='type', default="BRCA", help="cancer type: BRCA")
parser.add_argument("-n", dest='n_cluster', default=-1, help="cancer subtype number for cluster")
parser.add_argument("-lr", dest='learning_rate', type=float, default=0.005, help="")
parser.add_argument("-d", dest='latent_dim', type=int, default=256, help="")
parser.add_argument("-batch_size", dest='batch_size', type=int, default=64, help="")
parser.add_argument("-use_gpu", dest='use_gpu', type=bool, default=True, help="whether to use GPU")

args = parser.parse_args()
log.info(str(args))

if args.use_gpu and torch.cuda.is_available():
    use_gpu = True
else:
    log.info("use CPU for training")


def load_data(cancer_type, tensor=True):
    ldata = []

    omics_data_type = []
    omics_data_size = []
    for omics_type in ["miRNA", 'Mut', 'CNA', 'rna']:
        fea_save_file = f'../fea/{cancer_type}/{omics_type}.fea'

        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
            log.info(f"directly read data: {fea_save_file} : {df.shape}")

        else:
            log.info(f'{cancer_type} no {omics_type} data type')
            continue

        omics_data_type.append(omics_type)
        omics_data_size.append(df.shape[1])
        if tensor:
            ldata.append(torch.from_numpy(df.values).float())
        else:
            ldata.append(df)

    log.info(f"{args.type} {omics_data_type}: {omics_data_size}")
    return ldata, omics_data_type, omics_data_size


def run_train_WGME(cancer_type):
    assert args.type in list(cancer_dict.keys()) or args.n_cluster != -1, \
        "you should specify cluster number!"

    if args.n_cluster != -1:
        n_cluster = args.n_cluster
    else:
        n_cluster = cancer_dict[cancer_type]

    ldata, omics_data_type, omics_data_size = load_data(cancer_type, tensor=True)
    epochs = args.epochs

    if use_gpu:
        ldata = [ldata[i].cuda() for i in range(len(ldata))]

    # Initialize model
    model = SubtypeWGME(omics_data_type, omics_data_size, latent_dim=args.latent_dim)
    early_stopping = EarlyStopping(cancer_type=cancer_type, n_cluster=n_cluster, valid_data=ldata,
                                   max_stop_epoch=args.epochs)

    # Loss function
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    latent_dim = args.latent_dim
    batch_size = args.batch_size

    real = torch.ones((batch_size, 1)).float()
    fake = torch.zeros((batch_size, 1)).float()
    if use_gpu:
        real = real.cuda()
        fake = fake.cuda()
        model = model.cuda()
        mse_loss = mse_loss.cuda()
        bce_loss = bce_loss.cuda()

    loss = []
    start_time = time.time()
    for epoch in range(epochs + 1):
        X = []
        idx = np.arange(0, ldata[0].shape[0])
        np.random.shuffle(idx)
        for i, _ in enumerate(omics_data_size):
            tmp = ldata[i][idx[0:batch_size]]
            X.append(tmp)

        for i in range(2):
            latent_fake = model.encode(X).detach()
            latent_real = torch.randn(batch_size, latent_dim).float()
            if use_gpu:
                latent_real = latent_real.cuda()

            d_loss_real = bce_loss(model.disc(latent_real), real)  
            d_loss_fake = bce_loss(model.disc(latent_fake), fake)  

            d_loss = torch.add(d_loss_real, d_loss_fake) 
            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

        img_x = model.encoder.data_process(X)
        latent_fake, disc_res, con_x = model(X)

        loss0 = 0.
        for i, _ in enumerate(omics_data_size):
            loss0 += mse_loss(con_x[0][i], X[i])
        loss1 = mse_loss(con_x[1], img_x)
        g_loss = loss0 + loss1
        disc_loss = bce_loss(disc_res, real)
        g_loss += 0.01 * disc_loss

        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()
        if use_gpu:
            g_loss = g_loss.cpu()
        loss.append(g_loss.detach().numpy())

        if epoch % 5 == 0:
            early_stopping(epoch, model)
            log.info(f"current best p value : \033[1;31;43m{str(early_stopping.best_p_value)}\033[0m")
            if early_stopping.early_stop:
                print("Early stopping")
                log.info(f"total time: {time.time() - start_time}")
                return
            log.info(
                f"{epoch} loss0:{round(float(loss0.cpu()), 4)} loss1:{round(float(loss1.cpu()), 4)} ")

    vec = early_stopping.get_feature(early_stopping.valid_data, model)
    df_c = early_stopping.get_cluster(vec)
    early_stopping.save_feature(vec, df_c)
    log.info(f"total time: {time.time() - start_time}")


def run_biomarker_rf(cancer_type):
    ldata, omics_data_type, omics_data_size = load_data(cancer_type, tensor=False)
    X = pd.concat(ldata, axis=1)

    feature = list(X.columns)
    y = pd.read_csv(f'../results/{cancer_type}/{cancer_type}.SubtypeWGME', index_col=0)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, oob_score=True)
    model.fit(X.values, ravel(y.values))

    omics_type = []
    for type, num in zip(omics_data_type, omics_data_size):
        omics_type += [type for _ in range(num)]
    df_dict = {'score': model.feature_importances_, 'omics_type': omics_type}
    importance_df = pd.DataFrame(data=df_dict, index=feature)
    importance_df.sort_values('score', ascending=False, inplace=True)
    os.makedirs("../results/biomarker/randomforest", exist_ok=True)
    importance_df.to_csv(f"../results/biomarker/randomforest/{args.type}.score", index=True)


def run_biomarker_xgboost(cancer_type):
    ldata, omics_data_type, omics_data_size = load_data(cancer_type, tensor=False)
    X = pd.concat(ldata, axis=1)

    feature = list(X.columns)
    y = pd.read_csv(f'../results/{cancer_type}/{cancer_type}.SubtypeWGME', index_col=0)
    y['subtype'] = y['subtype'] - 1
    model = XGBClassifier()
    model.fit(X.values, ravel(y.values))

    omics_type = []
    for type, num in zip(omics_data_type, omics_data_size):
        omics_type += [type for _ in range(num)]
    df_dict = {'score': model.feature_importances_, 'omics_type': omics_type}
    importance_df = pd.DataFrame(data=df_dict, index=feature)
    importance_df.sort_values('score', ascending=False, inplace=True)
    os.makedirs("../results/biomarker/xgboost", exist_ok=True)
    importance_df.to_csv(f"../results/biomarker/xgboost/{args.type}.score", index=True)


def run():
    cancer_type = args.type
    out_file_path = '../results/' + cancer_type + '/'  # ./results/PACA/
    os.makedirs(out_file_path, exist_ok=True)

    if args.run_mode == 'SubtypeWGME':
        run_train_WGME(cancer_type)
    if args.run_mode == 'biomarker_rf':
        run_biomarker_rf(cancer_type)
    if args.run_mode == 'biomarker_xgb':
        run_biomarker_xgboost(cancer_type)


if __name__ == "__main__":
    print_environment_info()
    provide_determinism(0)
    run()

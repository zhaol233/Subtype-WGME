# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from numpy import ravel

from utils.myutils import print_environment_info, provide_determinism, cancer_dict
from utils.logger import log
from model import SubtypeWGME, ValidModel
import os

parser = argparse.ArgumentParser(description='SubtypeWGME v1.0')
parser.add_argument("-e",
                    dest='epochs',
                    type=int,
                    default=200,
                    help="number of max iterations")
parser.add_argument("-m",
                    dest='run_mode',
                    default="SubtypeWGME",
                    help="SubtypeWGME, biomarker_rf")

parser.add_argument("-t",
                    dest='type',
                    default="BRCA",
                    help="cancer type: BRCA")
parser.add_argument("-n",
                    dest='n_cluster',
                    default=-1,
                    help="cancer subtype number for cluster")

parser.add_argument("-lr",
                    dest='learning_rate',
                    type=float,
                    default=0.005,
                    help="")
parser.add_argument("-d", dest='latent_dim', type=int, default=256, help="")
parser.add_argument("-batch_size",
                    dest='batch_size',
                    type=int,
                    default=64,
                    help="")
parser.add_argument("-use_gpu",
                    dest='use_gpu',
                    type=bool,
                    default=True,
                    help="whether to use GPU")
parser.add_argument("--valid", action="store_true", help="train or valid")
parser.add_argument("-omics_list",
                    type=str,
                    default="miRNA Mut CNA rna",
                    help="which omics to use")
parser.add_argument("-region",
                    type=str,
                    default="all",
                    help="which region to use")

args = parser.parse_args()
log.info(str(args))

use_gpu = False
if args.use_gpu and torch.cuda.is_available():
    use_gpu = True
else:
    log.info("use CPU for training")


def load_data(cancer_type, tensor=True):
    train_data = []
    valid_data = []
    all_data = []
    omics_list = args.omics_list.split(' ')
    omics_data_type = []
    omics_data_size = []

    all_clinic = pd.read_csv(f"../results/{cancer_type}/clinic.csv",
                             index_col=0)
    train_clinic = pd.read_csv(f"../results/{cancer_type}/train_clinic.csv",
                               index_col=0)
    valid_clinic = pd.read_csv(f"../results/{cancer_type}/valid_clinic.csv",
                               index_col=0)

    for omics_type in omics_list:
        fea_save_file = f'../fea/{cancer_type}/{omics_type}.fea'
        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
            log.info(f"directly read data: {fea_save_file} : {df.shape}")
        else:
            log.info(f'{cancer_type} no {omics_type} data type')
            continue
        # coding and noncoding region split
        if args.region == 'coding':
            if omics_type == 'rna':
                df = df.T
                df['index'] = df.index
                df['pos'] = df['index'].apply(
                    lambda x: x.split('::')[0].split('.')[-1])
                df = df[df['pos'] == 'gencode']
                df.drop(labels=['index', 'pos'], axis=1, inplace=True)
                df = df.T
            elif omics_type in ['CNA', 'Mut']:
                df = df.T
                df['index'] = df.index
                df['pos'] = df['index'].apply(
                    lambda x: x.split('::')[0].split('.')[-1])
                df = df[df['pos'] == 'cds']
                df.drop(labels=['index', 'pos'], axis=1, inplace=True)
                df = df.T
        if args.region == 'noncoding':
            if omics_type == 'rna':
                df = df.T
                df['index'] = df.index
                df['pos'] = df['index'].apply(
                    lambda x: x.split('::')[0].split('.')[-1])
                df = df[df['pos'] != 'gencode']
                df.drop(labels=['index', 'pos'], axis=1, inplace=True)
                df = df.T
            elif omics_type in ['CNA', 'Mut']:
                df = df.T
                df['index'] = df.index
                df['pos'] = df['index'].apply(
                    lambda x: x.split('::')[0].split('.')[-1])
                df = df[df['pos'] != 'cds']
                df.drop(labels=['index', 'pos'], axis=1, inplace=True)
                df = df.T

        omics_data_type.append(omics_type)
        omics_data_size.append(df.shape[1])

        if tensor:
            all_data.append(torch.from_numpy(df.values).float())
            valid_data.append(
                torch.from_numpy(df.loc[valid_clinic.index, :].values).float())
            train_data.append(
                torch.from_numpy(df.loc[train_clinic.index, :].values).float())
        else:
            all_data.append(df)
            valid_data.append(df.loc[valid_clinic.index, :])
            train_data.append(df.loc[train_clinic.index, :])

    if len(all_data) == 0:
        log.info("no data found")
        exit(1)
    log.info(f"{args.type}  {omics_data_type}: {omics_data_size}")
    return all_data, all_clinic, omics_data_type, omics_data_size


def run_train_WGME(cancer_type):
    assert args.type in list(cancer_dict.keys(
    )) or args.n_cluster != -1, "you should specify cluster number!"

    if args.n_cluster != -1:
        n_cluster = args.n_cluster
    else:
        n_cluster = cancer_dict[cancer_type]
    # load data
    time0 = time.time()
    all_data, all_clinic, omics_data_type, omics_data_size = load_data(
        cancer_type)
    time1 = time.time()
    log.info(f"load date time: {time1 - time0}")
    epochs = args.epochs
    if use_gpu:
        all_data = [all_data[i].cuda() for i in range(len(all_data))]
        # valid_data = [valid_data[i].cuda() for i in range(len(valid_data))]
    # Initialize model
    valid_model = ValidModel(cancer_type=cancer_type,
                             n_cluster=n_cluster,
                             valid_data=all_data,
                             clinic_data=all_clinic)

    if args.valid:
        log.info("just inference")
        time_start = time.time()
        vmodel = SubtypeWGME(omics_data_type,
                             omics_data_size,
                             latent_dim=args.latent_dim)
        vmodel = vmodel.cuda()
        model_path = f"../results/model_all/{cancer_type}.pt"
        if args.region != 'all':
            model_path = f"../results/model_{args.region}/{cancer_type}.pt"
        elif args.omics_list != 'miRNA Mut CNA rna':  # only single omics
            model_path = f"../results/model_{args.omics_list}/{cancer_type}.pt"
        log.info(f"valid model path: {model_path}")
        vmodel.load_state_dict(torch.load(model_path))
        vmodel.eval()
        valid_model.getp(vmodel)
        time_end = time.time()
        log.info(f"inference time: {time_end - time_start}")
        exit(0)

    model = SubtypeWGME(omics_data_type,
                        omics_data_size,
                        latent_dim=args.latent_dim)

    # Loss function
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    latent_dim = args.latent_dim
    batch_size = args.batch_size
    samples_n = all_data[0].shape[0]

    real = torch.ones((batch_size, 1)).float()
    fake = torch.zeros((batch_size, 1)).float()
    if use_gpu:
        real = real.cuda()
        fake = fake.cuda()
        model = model.cuda()
        mse_loss = mse_loss.cuda()
        bce_loss = bce_loss.cuda()

    loss = []
    time2 = time.time()
    log.info(f"model init time: {time2 - time1}")
    print(epochs, batch_size)
    for epoch in range(epochs + 1):
        X = []
        idx = np.arange(0, samples_n)
        np.random.shuffle(idx)  # random choose batch_size samples
        for i, _ in enumerate(omics_data_size):
            tmp = all_data[i][idx[0:batch_size]]
            X.append(tmp)
        loss_real = 0.0
        loss_fake = 0.0
        for i in range(2):
            latent_fake = model.encode(X).detach()
            latent_real = torch.randn(batch_size, latent_dim).float()
            if use_gpu:
                latent_real = latent_real.cuda()
            d_loss_real = bce_loss(model.disc(latent_real), real)
            d_loss_fake = bce_loss(model.disc(latent_fake), fake)

            d_loss = torch.add(d_loss_real, d_loss_fake)  # disc loss

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()
            loss_real = round(float(d_loss_real.cpu()), 4)
            loss_fake = round(float(d_loss_fake.cpu()), 4)

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
            valid_model(epoch, model)
            log.info(
                f"epoch: {epoch} loss0:{round(float(loss0.cpu()), 4)} loss1:{round(float(loss1.cpu()), 4)} loss_real: {loss_real} loss fake: {loss_fake}"
            )
    valid_model.save(model)
    time3 = time.time()
    log.info(f"model train and save time: {time3 - time2}")


def run_biomarker_rf(cancer_type):
    train_clinic = pd.read_csv(f"../results/{cancer_type}/train_clinic.csv",
                               index_col=0)
    valid_clinic = pd.read_csv(f"../results/{cancer_type}/valid_clinic.csv",
                               index_col=0)
    train_data = []
    valid_data = []
    omics_data_type = []
    omics_data_size = []
    omics_list = args.omics_list.split(' ')
    for omics_type in omics_list:
        fea_save_file = f'../fea/{cancer_type}/{omics_type}.fea'
        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
        else:
            log.info(f'{cancer_type} no {omics_type} data type')
            continue
        omics_data_type.append(omics_type)
        omics_data_size.append(df.shape[1])
        valid_data.append(df.loc[valid_clinic.index, :])
        train_data.append(df.loc[train_clinic.index, :])
    label = pd.read_csv(f'../results/{cancer_type}/{cancer_type}.SubtypeWGME',
                        index_col=0)
    # data = load_data(cancer_type, tensor=False)
    train_X = pd.concat(train_data, axis=1)
    feature = list(train_X.columns)  # feature name

    train_y = label.loc[train_clinic.index, :]
    valid_X = pd.concat(valid_data, axis=1)
    valid_y = label.loc[valid_clinic.index, :]
    is_binary = True if np.unique(
        train_y).shape[0] == 2 else False  # special 'binary' case

    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=10,
                                   random_state=0,
                                   oob_score=True)
    model.fit(train_X.values, ravel(train_y.values))

    predictions = model.predict_proba(train_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc_t = roc_auc_score(train_y.values, predictions, multi_class='ovr')

    predictions = model.predict_proba(valid_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc = roc_auc_score(valid_y.values, predictions, multi_class='ovr')

    log.info(
        f'The baseline auc score on the train valid set is {auc_t} {auc}.'.
        format(auc_t, auc))

    omics_type = []
    for type, num in zip(omics_data_type, omics_data_size):
        omics_type += [type for _ in range(num)]
    df_dict = {'score': model.feature_importances_, 'omics_type': omics_type}
    importance_df = pd.DataFrame(data=df_dict, index=feature)
    importance_df.sort_values('score', ascending=False, inplace=True)
    os.makedirs("../results/biomarker/randomforest", exist_ok=True)
    importance_df.to_csv(
        f"../results/biomarker/randomforest/{args.type}.score", index=True)


def run_biomarker_xgboost(cancer_type):
    train_clinic = pd.read_csv(f"../results/{cancer_type}/train_clinic.csv",
                               index_col=0)
    valid_clinic = pd.read_csv(f"../results/{cancer_type}/valid_clinic.csv",
                               index_col=0)
    train_data = []
    valid_data = []
    omics_data_type = []
    omics_data_size = []
    omics_list = args.omics_list.split(' ')
    for omics_type in omics_list:
        fea_save_file = f'../fea/{cancer_type}/{omics_type}.fea'
        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
        else:
            log.info(f'{cancer_type} no {omics_type} data type')
            continue
        omics_data_type.append(omics_type)
        omics_data_size.append(df.shape[1])
        valid_data.append(df.loc[valid_clinic.index, :])
        train_data.append(df.loc[train_clinic.index, :])
    label = pd.read_csv(f'../results/{cancer_type}/{cancer_type}.SubtypeWGME',
                        index_col=0)
    train_X = pd.concat(train_data, axis=1)
    feature = list(train_X.columns)  # feature name

    train_y = label.loc[train_clinic.index, :]
    train_y = train_y - 1
    valid_X = pd.concat(valid_data, axis=1)
    valid_y = label.loc[valid_clinic.index, :]
    valid_y = valid_y - 1
    is_binary = True if np.unique(
        train_y).shape[0] == 2 else False  # special 'binary' case

    model = XGBClassifier()
    model.fit(train_X.values, ravel(train_y.values))
    predictions = model.predict_proba(train_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc_t = roc_auc_score(train_y.values, predictions, multi_class='ovr')

    predictions = model.predict_proba(valid_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc = roc_auc_score(valid_y.values, predictions, multi_class='ovr')
    log.info(
        f'The baseline auc score on the train valid set is {auc_t} {auc}.'.
        format(auc_t, auc))

    omics_type = []
    for type, num in zip(omics_data_type, omics_data_size):
        omics_type += [type for _ in range(num)]
    df_dict = {'score': model.feature_importances_, 'omics_type': omics_type}
    importance_df = pd.DataFrame(data=df_dict, index=feature)
    importance_df.sort_values('score', ascending=False, inplace=True)
    os.makedirs("../results/biomarker/xgboost", exist_ok=True)
    importance_df.to_csv(f"../results/biomarker/xgboost/{args.type}.score",
                         index=True)
    print(
        f"save feature importance to results/biomarker/xgboost/{cancer_type}.score. done."
    )


def run_biomarker_lgbm(cancer_type):
    train_clinic = pd.read_csv(f"../results/{cancer_type}/train_clinic.csv",
                               index_col=0)
    valid_clinic = pd.read_csv(f"../results/{cancer_type}/valid_clinic.csv",
                               index_col=0)
    train_data = []
    valid_data = []
    omics_data_type = []
    omics_data_size = []
    omics_list = args.omics_list.split(' ')
    for omics_type in omics_list:
        fea_save_file = f'../fea/{cancer_type}/{omics_type}.fea'
        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
        else:
            log.info(f'{cancer_type} no {omics_type} data type')
            continue
        omics_data_type.append(omics_type)
        omics_data_size.append(df.shape[1])
        valid_data.append(df.loc[valid_clinic.index, :])
        train_data.append(df.loc[train_clinic.index, :])
    label = pd.read_csv(f'../results/{cancer_type}/{cancer_type}.SubtypeWGME',
                        index_col=0)
    # data = load_data(cancer_type, tensor=False)
    train_X = pd.concat(train_data, axis=1)
    feature = list(train_X.columns)  # feature name

    train_y = label.loc[train_clinic.index, :]
    train_y = train_y - 1
    valid_X = pd.concat(valid_data, axis=1)
    valid_y = label.loc[valid_clinic.index, :]
    valid_y = valid_y - 1
    is_binary = True if np.unique(
        train_y).shape[0] == 2 else False  # special 'binary' case

    model = LGBMClassifier()
    model.fit(train_X.values, ravel(train_y.values))
    predictions = model.predict_proba(train_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc_t = roc_auc_score(train_y.values, predictions, multi_class='ovr')

    predictions = model.predict_proba(valid_X.values)
    if is_binary:
        predictions = np.argmax(predictions, axis=1)
    auc = roc_auc_score(valid_y.values, predictions, multi_class='ovr')
    log.info(
        f'The baseline auc score on the train valid set is {auc_t} {auc}.'.
        format(auc_t, auc))

    omics_type = []
    for type, num in zip(omics_data_type, omics_data_size):
        omics_type += [type for _ in range(num)]
    df_dict = {'score': model.feature_importances_, 'omics_type': omics_type}
    importance_df = pd.DataFrame(data=df_dict, index=feature)
    importance_df.sort_values('score', ascending=False, inplace=True)
    os.makedirs("../results/biomarker/lgbm", exist_ok=True)
    importance_df.to_csv(f"../results/biomarker/lgbm/{args.type}.score",
                         index=True)
    print(
        f"save feature importance to results/biomarker/lgbm/{cancer_type}.score. done."
    )


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
    if args.run_mode == 'biomarker_lgbm':
        run_biomarker_lgbm(cancer_type)


if __name__ == "__main__":
    print_environment_info()
    provide_determinism(0)
    run()

# -*- coding: utf-8 -*-

import math
import os

from numpy import ravel
from sklearn.metrics import silhouette_score

from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
import pandas as pd

from sklearn import mixture

import numpy as np
from itertools import combinations
import bisect

from utils.logger import log
from utils.myutils import cancer_dict

root_path = "/content/drive/MyDrive/ZL/paper1"


class ConsensusCluster:
    def __init__(self, cluster, L, K, H, resample_proportion=0.8):
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion):
        ids = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False)
        return ids, data[ids, :]

    def fit(self, data):
        Mk = np.zeros((self.K_ - self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],) * 2)
        for k in range(self.L_, self.K_):
            i_ = k - self.L_
            for h in range(self.H_):
                ids, dt = self._internal_resample(data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(dt)
                ids_sorted = np.argsort(Mh)
                sorted_ = Mh[ids_sorted]
                for i in range(k):
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = ids_sorted[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                ids_2 = np.array(list(combinations(ids, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is + 1e-8
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1
            Is.fill(0)
        self.Mk = Mk
        self.Ak = np.zeros(self.K_ - self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = sum(h * (b - a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        self.deltaK = np.array([(Ab - Aa) / Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_ - 1))])
        self.bestK = np.argmax(self.deltaK) + \
                     self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1 - self.Mk[self.bestK - self.L_])

    def predict_data(self, data):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


def gmm_model(n_clusters=28):
    model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0, reg_covar=1e-3)
    return model


def gmm(n_cluster, cancer_type, fea_df=None, save=True, method='SubtypeWGME'):
    cluster_file = root_path + f"/results/{cancer_type}/{cancer_type}.{method}"
    fea_tmp_file = root_path + f'/fea/{cancer_type}/{cancer_type}.fea'
    if method == 'SubtypeGAN':
        fea_tmp_file = root_path + f'/fea/{cancer_type}/{cancer_type}.SubtypeGAN'
    if fea_df is not None:
        df_data = fea_df
    else:
        df_data = pd.read_csv(fea_tmp_file, header=0, index_col=0)
    model = gmm_model(n_cluster)
    subtype = model.fit_predict(df_data.values) + 1
    df_cluster = pd.DataFrame(index=df_data.index, columns=['subtype'], data=subtype)

    if save:
        # log.info(f"save cluster file to : {cluster_file}")
        df_cluster.to_csv(cluster_file, header=True, index=True)
    return df_cluster


def proba(n_cluster, type):
    cluster_file = root_path + f"/results/{type}/{type}.proba"
    fea_tmp_file = root_path + f'/fea/{type}/{type}.fea'
    df_data = pd.read_csv(fea_tmp_file, header=0, index_col=0)
    model = gmm_model(n_cluster)
    model.fit(df_data.values)

    df = model.predict_proba(df_data.values)
    pd.DataFrame(df, index=df_data.index).to_csv(cluster_file, header=True, index=True, sep=',')


def get_p(cancer_type, df_c=None, df_l=None, method='SubtypeWGME'):
    clinical_path = root_path + f"/results/{cancer_type}/clinic.csv"
    cluster_file = root_path + f"/results/{cancer_type}/{cancer_type}.{method}"
    if df_c is None:
        df_cluster = pd.read_csv(cluster_file, header=0, index_col=0)
    else:
        df_cluster = df_c
    if df_l is None:
        df_clinic = pd.read_csv(clinical_path, sep=',', header=0, index_col=0)
    else:
        df_clinic = df_l
    df_clinic = df_clinic.join(df_cluster)
    kmf = KaplanMeierFitter()
    for name, grouped_df in df_clinic.groupby('subtype'):
        kmf.fit(grouped_df["days"], grouped_df["status"], label=name)
        # kmf.plot_survival_function()
    result = multivariate_logrank_test(df_clinic['days'], df_clinic['subtype'], df_clinic['status'])
    pvalue = round(-math.log10(result.p_value), 3)
    return pvalue


def get_p_assign(cancer_type='all', method="SubtypeWGME"):
    if cancer_type != 'all':
        if method in ['SubtypeWGME', 'SubtypeGAN']:
            X = gmm(cancer_dict[cancer_type], cancer_type, fea_df=None, save=True, method=method)
        else:
            subtype_file = root_path + f"/results/{cancer_type}/{cancer_type}.{method}"
            if os.path.isfile(subtype_file):
                X = pd.read_csv(subtype_file)
            else:
                X = None
        if X:
            log.info(f"{cancer_type} {method} p value: {get_p(cancer_type, X, method=method)}")
        else:
            log.info(f"{method} unsupported {cancer_type} ")
    else:
        os.makedirs(root_path + f"/results/pvalue", exist_ok=True)
        res_file = root_path + f"/results/pvalue/{method}_assigned_pvalue.csv"
        x = pd.DataFrame(list(cancer_dict.keys()))
        p_value = []
        for cancer in list(cancer_dict.keys()):
            # print(cancer_dict[cancer], cancer)
            if method in ['SubtypeWGME', 'SubtypeGAN']:
                X = gmm(cancer_dict[cancer], cancer, fea_df=None, save=True, method=method)
            else:
                subtype_file = root_path + f"/results/{cancer}/{cancer}.{method}"
                if os.path.isfile(subtype_file):
                    X = pd.read_csv(subtype_file)
                else:
                    X = None
            if X is not None:
                p_value.append(get_p(cancer_type=cancer, df_c=X, df_l=None, method=method))
            else:
                p_value.append(0)

        x.insert(x.shape[1], 'pvalue', p_value)
        log.info(x['pvalue'].values)
        log.info(x['pvalue'].median())
        log.info(f'save results file to {res_file}')
        x.to_csv(res_file, index=True, header=True, sep=',', index_label=0)


def cc(cancer_type='all', method='euclidean'):
    #         {"CLLE": 6, "ESAD": 2, "MALY": 3, "OV": 3, "PACA": 6, "PAEN": 6, "RECA": 4,"BRCA": 4}
    global X
    K1_dict = {"CLLE": 3, "ESAD": 2, "MALY": 3, "OV": 3, "PACA": 3, "PAEN": 3, "RECA": 2, "BRCA": 3}

    K2_dict = {"CLLE": 8, "ESAD": 6, "MALY": 6, "OV": 8, "PACA": 8, "PAEN": 8, "RECA": 6, "BRCA": 8}
    print(K2_dict)
    best_k_p_value = []
    best_k = []
    x = pd.DataFrame(list(cancer_dict))
    res_file = root_path + f"/results/pvalue/cc_pvalue.csv"

    if cancer_type != 'all':
        cancer = cancer_type
        clinical_path = root_path + f"/results/{cancer}/clinic.csv"
        df_clinic = pd.read_csv(clinical_path, sep=',', header=0, index_col=0)
        fea_tmp_file = root_path + f'/fea/{cancer}/{cancer}.fea'
        fea_df = pd.read_csv(fea_tmp_file, header=0, index_col=0)
        ccluster = ConsensusCluster(gmm_model, K1_dict[cancer], K2_dict[cancer], 5)
        ccluster.fit(fea_df.values)
        X = gmm(ccluster.bestK, cancer, fea_df, save=True, method='cc')
        log.info(
            f" {cancer} auto cluster num {ccluster.bestK}  p_value {get_p(cancer, df_clinic, X)} {cancer_dict[cancer]}")
        return

    for cancer in list(cancer_dict.keys()):
        clinical_path = root_path + f"/results/{cancer}/clinic.csv"
        df_clinic = pd.read_csv(clinical_path, sep=',', header=0, index_col=0)
        fea_tmp_file = root_path + f'/fea/{cancer}/{cancer}.fea'
        fea_df = pd.read_csv(fea_tmp_file, header=0, index_col=0)

        ccluster = ConsensusCluster(gmm_model, K1_dict[cancer], 10, 10)
        ccluster.fit(fea_df.values)
        X = gmm(ccluster.bestK, cancer, fea_df, save=True, method='cc')

        best_k.append(ccluster.bestK)
        best_k_p_value.append(get_p(cancer, df_l=df_clinic, df_c=X))

    accuracy = [best_k[i] == list(cancer_dict.values())[i] for i in range(len(cancer_dict))].count(True) / len(
        cancer_dict)
    print(best_k)
    print(best_k_p_value)
    print('自动推荐中位数', np.median(best_k_p_value))
    print(f'{method} auto accuracy {accuracy * 100}%')
    # x.to_csv(res_file, index=False, header=True, sep=',')


def get_proba(cancer_type='all'):
    if cancer_type != 'all':
        proba(cancer_dict[cancer_type], cancer_type)
        log.info(f"{cancer_type} get proba file finished ")
    else:
        pass

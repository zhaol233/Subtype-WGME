# -*- coding: utf-8 -*-
# @Time    : 2022/9/19 12:44
# @Author  : zhaoliang
# @Description: get clinical information omics data for each cancer
import os

import numpy as np
import pandas as pd
from logger import log
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

os.makedirs('../results', exist_ok=True)


def get_clinical_file():
    surv_path = "../data/pcawg_donor_clinical_August2016_v9.csv"
    clinic_params = ['# donor_unique_id', 'project_code', 'donor_vital_status', 'donor_survival_time',
                     'donor_interval_of_last_followup', 'donor_sex', 'donor_age_at_diagnosis']
    df_clinic_all = pd.read_csv(surv_path, header=0, sep=',', usecols=clinic_params)  # 2834
    df_clinic_all = df_clinic_all[df_clinic_all['donor_vital_status'].notnull()]  # (2665, 7)
    df_clinic_all['status'] = np.where(df_clinic_all['donor_vital_status'] == 'deceased', 1, 0)
    df_clinic_all['days'] = df_clinic_all.apply(
        lambda r: r['donor_survival_time'] if r['donor_vital_status'] == 1 else r[
            'donor_interval_of_last_followup'], axis=1)

    df_clinic_all = df_clinic_all[df_clinic_all['days'].notnull()]
    df_clinic_all['acronym'] = df_clinic_all['project_code'].apply(lambda x: str(x).split('-')[0])
    df_clinic_all = df_clinic_all.loc[:,
                    ['# donor_unique_id', 'status', 'days', 'donor_sex', 'donor_age_at_diagnosis', 'acronym']]
    df_clinic_all.to_csv("../results/sample_with_survival_data.csv", index=False)


def get_omics():
    nb_line = 0
    omics_status_dict = {"miRNA": [], "rna": [], "Mut": [], "CNA": []}
    cancer_pool = {"OV": [], "PAEN": [], "RECA": [], "CLLE": [], "ESAD": [], "MALY": [], "PACA": [], 'BRCA': []}

    df_clinic = pd.read_csv("../results/sample_with_survival_data.csv", index_col=0)

    for omics_type in list(omics_status_dict.keys()):
        nb_line += 1
        raw_file = f"../data/{omics_type}.gz"
        log.info(f"--{nb_line}:--{omics_type}-------------------- read {raw_file}----")
        df_new = pd.read_csv(raw_file, index_col=0)
        df_new = df_new.T

        df_new.fillna(0, inplace=True)
        if omics_type in ['miRNA', 'rna']:
            df_new = np.log2(df_new + 1)

        for cancer_type in list(cancer_pool.keys()):
            os.makedirs(f'../results/{cancer_type}', exist_ok=True)
            # print(cancer_type)
            # 1. get samples and clinical for each cancer
            if nb_line == 1:
                clinic_save_path = f"../results/{cancer_type}/clinic.csv"
                if os.path.isfile(clinic_save_path):
                    sample_name = list(pd.read_csv(clinic_save_path, index_col=0).index)
                    cancer_pool[cancer_type].append(sample_name)
                else:
                    df_clinic_each = df_clinic.loc[df_clinic['acronym'] == cancer_type, ::]
                    sample_name = list(df_clinic_each.index)
                    feature_sample = list(df_new.index)
                    sample_name = list(set(sample_name).intersection(set(feature_sample)))
                    df_clinic.loc[sample_name, :].to_csv(clinic_save_path, index=True)
                    cancer_pool[cancer_type].append(sample_name)

            # print(cancer_pool[cancer_type][0])
            df_cancer = df_new.loc[cancer_pool[cancer_type][0], :]
            # 2. preprocessing omics data
            selector = VarianceThreshold(threshold=0.2)
            try:
                selector.fit(df_cancer)
                df_cancer = df_cancer.loc[:, selector.get_support()]
                omics_status_dict[omics_type].append(df_cancer.shape[1])
            except:
                log.info(f"{cancer_type} has no {omics_type} data after VarianceThreshold selector")
                omics_status_dict[omics_type].append(0)

                continue
            else:
                pass
            # save fea without norm
            os.makedirs(f"../fea/{cancer_type}", exist_ok=True)
            fea_save_file = f"../fea/{cancer_type}/{omics_type}.rawfea"
            df_cancer.to_csv(fea_save_file, index=True, header=True, sep=',')

            scaler = preprocessing.StandardScaler()
            mat = scaler.fit_transform(df_cancer.values.astype(float))
            df_cancer.iloc[::, ::] = mat

            # 3. save don't change datasets
            fea_save_file = f"../fea/{cancer_type}/{omics_type}.fea"
            df_cancer.to_csv(fea_save_file, index=True, header=True, sep=',')

    # df = pd.DataFrame(omics_status_dict)
    # df.index = list(cancer_pool.keys())
    # df.to_csv("../results/omics_len.csv", index=True)


if __name__ == "__main__":
    get_clinical_file()
    get_omics()

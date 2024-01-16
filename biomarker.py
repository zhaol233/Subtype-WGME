import pandas as pd
import math
import os
import sys
input_path = "/home/ec2-user/SageMaker/paper1/results"
method = sys.argv[2]
def get_p(df):
    from lifelines.statistics import multivariate_logrank_test
    result = multivariate_logrank_test(df['days'], df['type'], df['status'])
    pvalue = round(-math.log10(result.p_value), 3)
    return pvalue

def get_top_biomarker_score(cancer, save=True):
    df = pd.read_csv(f"{input_path}/biomarker/{method}/{cancer}.score", index_col=0)
    df['index'] = df.index
    score_all = df['score'].sum()
    df['score'] = df['score'].apply(lambda x: x / score_all)

    df = df[~df['index'].str.contains("enhancer")]
    res_df = pd.DataFrame()
    res_omics = []
    omics_all = ['miRNA', 'rna', 'CNA', 'Mut']
    for omics in omics_all:
        s = df[df['omics_type'] == omics].iloc[0:50]
        if omics == 'rna':
            s = s[s['index'].str.startswith('gencode')]
            if s.shape[0] == 0:
                continue
            else:
                s['index'] = s['index'].apply(lambda x: x.split("::")[1])
                res_df = pd.concat([res_df, s.iloc[0:50, :]], axis=0)
                res_omics.append(omics)
        if omics in ['CNA', 'Mut']:
            s = s[s['index'].str.contains('gencode')]
            if s.shape[0] == 0:
                continue
            s['index'] = s['index'].apply(
                lambda x: x.split("::")[2] + '(' + x.split("::")[0].split('.')[1] + ')')
            res_df = pd.concat([res_df, s.iloc[0:50, :]], axis=0)
            res_omics.append(omics)

        if omics == 'miRNA':
            if s.shape[0] == 0:
                continue
            s['index'] = s['index'].apply(lambda x: str(x)[4:])
            res_df = pd.concat([res_df, s.iloc[0:30, :]], axis=0)
            res_omics.append(omics)
    path =  f"{input_path}/biomarker/{method}/top50/{cancer}"
    os.makedirs(path, exist_ok=True)

    res_df.sort_values(by=['score'], ignore_index=False, ascending=False, inplace=True)
    res_df = res_df.iloc[0:50]
    res_df.to_csv(path + "/score.csv", index=True)
    return res_df, res_omics

def get_top_biomarker_rawfea(cancer_type, save=True):
    df_score = pd.read_csv( f"{input_path}/biomarker/{method}/top50/{cancer_type}/score.csv",
                           index_col=0,
                           header=0)
    ldata = []
    omics_all = ['miRNA', 'rna', 'CNA', 'Mut']
    for omics_type in omics_all:
        fea_save_file = f'/home/ec2-user/SageMaker/paper1/fea/{cancer_type}/{omics_type}.rawfea'
        if os.path.isfile(fea_save_file):
            df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
            df.columns = [i + omics_type for i in df.columns]
        else:
            continue
        ldata.append(df)

    df_fea = pd.concat(ldata, axis=1)

    needed_fea = [i + j for i, j in zip(df_score.index, df_score['omics_type'])]

    df_fea = df_fea.loc[:, needed_fea]
    df_fea.columns = df_score['index']
    path = f"{input_path}/biomarker/biomarker_rawfea"
    os.makedirs(path, exist_ok=True)
    df_fea.to_csv(path + f"/{cancer_type}.csv", index=True)
    return df_fea

def get_p_for_feature(cancer):
    clinic_file =  f"{input_path}/{cancer}/valid_clinic.csv"
    fea_file = f"{input_path}/biomarker/biomarker_rawfea/{cancer}.csv"
    score_file = f"{input_path}/biomarker/{method}/top50/{cancer}/score.csv"
    df_score = pd.read_csv(score_file, index_col=0)
    fullname = list(df_score.index)
    df_score.reset_index()
    df_score.set_index('index', inplace=True)
    df_fea = pd.read_csv(fea_file, index_col=0)
    cancer_l = list(df_fea.columns)
    df_clinic = pd.read_csv(clinic_file, index_col=0)  # 5
    df = pd.concat([df_clinic, df_fea], axis=1)
    df['days'] = df['days'] / 365
    pvalue = []
    for i in range(len(cancer_l)):
        fea = cancer_l[i]
        m = df[fea].median()
        df['type'] = df[fea].apply(lambda x: "high" if x > m else "low")
        pvalue.append(get_p(df))
    df_score.insert(2,'fullname',fullname)
    df_score.insert(2, 'pvalue', pvalue)
    df_score.to_csv(f'{input_path}/biomarker/{method}/top50/{cancer}/feature_p_value.csv',
                    index=True)

if __name__ == '__main__':
    cancer = sys.argv[1]
    print(cancer)
    get_top_biomarker_score(cancer)
    get_top_biomarker_rawfea(cancer)
    get_p_for_feature(cancer)

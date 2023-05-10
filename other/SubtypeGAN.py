import argparse
import bisect
import os
import random
import sys
import time
from itertools import combinations
import collections
from os.path import splitext, basename, isfile
# from util import *


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import mixture
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, concatenate, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt

# log_path = "./run.log"
# log = MyLogging(log_path).get_logger()


def set_global_determinism(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)


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
            self.Ak[i] = np.sum(h * (b - a)
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


class GeLU(Activation):
    def __init__(self, activation, **kwargs):
        super(GeLU, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'gelu': GeLU(gelu)})


class AE():
    def __init__(self, X_shape, n_components, epochs=100):
        self.epochs = epochs
        sample_size = X_shape[0]
        self.batch_size = 16
        sample_size = X_shape[0]
        self.epochs = 30
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        z = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        ae = Model(input, output)
        encoder = Model(input, z)
        ae_loss = mse(input, output)
        ae.add_loss(ae_loss)
        ae.compile(optimizer=Adam())
        print(len(ae.layers))
        print(ae.count_params())
        ae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return encoder.predict(X)


class VAE():
    def __init__(self, X_shape, n_components, epochs=100):
        self.epochs = epochs
        self.batch_size = 16
        sample_size = X_shape[0]
        self.epochs = 30
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), seed=0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        z_mean = Dense(encoding_dim)(encoded)
        z_log_var = Dense(encoding_dim)(encoded)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        vae = Model(input, output)
        encoder = Model(input, z)
        reconstruction_loss = mse(input, output)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        print(len(vae.layers))
        print(vae.count_params())
        vae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return encoder.predict(X)


class SubtypeHGAN():
    def __init__(self, datasets, n_latent_dim, weight=0.001, model_path='SubtypeHGAN.h5', epochs=100, batch_size=64):
        self.latent_dim = n_latent_dim
        optimizer = Adam()
        self.n = len(datasets)
        self.epochs = epochs
        self.batch_size = batch_size
        sample_size = 0
        if self.n > 1:
            sample_size = datasets[0].shape[0]

        print("total sample size: ",sample_size)
        self.epochs = 30 * batch_size
        self.shape = []
        self.weight = [0.25, 0.25, 0.25, 0.25]
        self.disc_w = 1e-4
        self.model_path = model_path
        input = []
        loss = []
        loss_weights = []
        output = []
        for i in range(self.n):
            self.shape.append(datasets[i].shape[1])
            loss.append('mse')
        loss.append('binary_crossentropy')

        ############ discriminator ############
        self.decoder, self.disc = self.build_decoder_disc()
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        ############# GAN ##############
        self.encoder = self.build_encoder()
        for i in range(self.n):
            input.append(Input(shape=(self.shape[i],)))
            loss_weights.append((1 - self.disc_w) * self.weight[i])
        loss_weights.append(self.disc_w)
        z_mean, z_log_var, z = self.encoder(input)
        output = self.decoder(z)   
        self.gan = Model(input, output,name="HGAN")

        
        self.gan.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.disc.summary())
        print(self.gan.summary())
        return

    def build_encoder(self):
        def sampling(args):
            z_mean, z_log_var = args
            return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(K.shape(z_mean), seed=0)

        encoding_dim = self.latent_dim
        X = []
        dims = []
        denses = []
        for i in range(self.n):
            X.append(Input(shape=(self.shape[i],)))
            dims.append(int(encoding_dim * self.weight[i]))    #  100 * 0.25 = 25
        for i in range(self.n):
            denses.append(Dense(dims[i])(X[i]))    # 64 组学数据单独降到固定比例的维度，每个都是25维，denses 四个输出(64,25)
        if self.n > 1:
            merged_dense = concatenate(denses, axis=-1)    #denses 四个输出concat     100维度 (64,100)
        else:
            merged_dense = denses[0] 

        print("merge_dense.shape",merged_dense.shape)  #(64, 100)

        model = BatchNormalization()(merged_dense)
        model = Activation('gelu')(model)
        model = Dense(encoding_dim)(model)    #64  100维度

        z_mean = Dense(encoding_dim)(model)
        z_log_var = Dense(encoding_dim)(model)

        # print("z_mean.shape",z_mean.shape)  # (16,100)

        z = Lambda(function=sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        return Model(X, [z_mean, z_log_var, z],name="encoder")    # 第三个输出 16，100

    def build_decoder_disc(self):
        denses = []
        X = Input(shape=(self.latent_dim,))
        model = Dense(self.latent_dim)(X)
        model = BatchNormalization()(model)
        model = Activation('gelu')(model)
        
        for i in range(self.n):
            denses.append(Dense(self.shape[i])(model))

        dec = Dense(1, activation='sigmoid')(model)  # disc 在 还原前计算损失
        denses.append(dec)
        m_decoder = Model(X, denses,name='decoder')  # decoder 还原四个输出 + dec分类损失
        m_disc = Model(X, dec,name="discriminator")

        return m_decoder, m_disc

    def build_disc(self):
        X = Input(shape=(self.latent_dim,))
        dec = Dense(1, activation='sigmoid', kernel_initializer="glorot_normal")(X)
        output = Model(X, dec)
        return output

    def train(self, X_train, bTrain=True):
        model_path = self.model_path
        log_file = "./run.log"
        fp = open(log_file, 'w')
        if bTrain:
            # GAN
            valid = np.ones((self.batch_size, 1))   # 真
            fake = np.zeros((self.batch_size, 1))   # 假
            loss = []
            for epoch in range(self.epochs):
                #  Train Discriminator
                data = []

                idx = np.random.randint(0, X_train[0].shape[0], self.batch_size)
                for i in range(self.n):
                    data.append(X_train[i][idx])

                latent_fake = self.encoder.predict(data)[2]
                 
                latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))

                d_loss_real = self.disc.train_on_batch(latent_real, valid)
                d_loss_fake = self.disc.train_on_batch(latent_fake, fake)   # loss
                
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                outs = data + [valid]  
              
                #  Train Encoder_GAN
                g_loss = self.gan.train_on_batch(data, outs)
            #     loss.append(g_loss)
            # plt.plot(loss,[i for i in range(self.epochs)],'g--',label='loss')
            # plt.title('loss')
            # plt.xlabel('eposh')
            # plt.ylabel('loss')
            # plt.legend()
            # plt.show()
                # fp.write("%f\t%f\n" % (g_loss[0], d_loss[0]))
            fp.close()
            # self.encoder.save(model_path)
        else:
            self.encoder = load_model(model_path)

        mat = self.encoder.predict(X_train)[0]
        return mat


class SubtypeHGAN_API(object):
    def __init__(self, model_path='./model/', epochs=200, weight=0.001):
        self.model_path = model_path
        self.score_path = './score/'
        self.epochs = epochs
        self.batch_size = 16
        self.weight = weight

    # feature extract
    def feature_gan(self, datasets, index=None, n_components=100, b_decomposition=True, weight=0.001):
        if b_decomposition:
            X = self.encoder_gan(datasets, n_components)
            print("X:::",X.shape)
            fea = pd.DataFrame(data=X, index=index, columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = np.concatenate(datasets)
        print("feature extract finished!")
        return fea

    def feature_vae(self, df_ori, n_components=100, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_vae(df_ori, n_components)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        print("feature extract finished!")
        return fea

    def feature_ae(self, df_ori, n_components=100, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_ae(df_ori, n_components)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        print("feature extract finished!")
        return fea

    def impute(self, X):
        X.fillna(X.mean())
        return X

    def encoder_gan(self, ldata, n_components=100):
        egan = SubtypeHGAN(ldata, n_components, self.weight, self.model_path, self.epochs, self.batch_size)
        return egan.train(ldata)

    def encoder_vae(self, df, n_components=100):
        vae = VAE(df.shape, n_components, self.epochs)
        return vae.train(df)

    def encoder_ae(self, df, n_components=100):
        ae = AE(df.shape, n_components, self.epochs)
        return ae.train(df)

    def tsne(self, X):
        model = TSNE(n_components=2)
        return model.fit_transform(X)

    def pca(self, X):
        fea_model = PCA(n_components=200)
        return fea_model.fit_transform(X)

    def gmm(self, n_clusters=28):
        model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
        return model

    def kmeans(self, n_clusters=28):
        model = KMeans(n_clusters=n_clusters, random_state=0)
        return model

    def spectral(self, n_clusters=28):
        model = SpectralClustering(n_clusters=n_clusters, random_state=0)
        return model

    def hierarchical(self, n_clusters=28):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='SubtypeHGAN v1.0')
    parser.add_argument("-i", dest='file_input', default="./input/input.list",
                        help="file input")
    parser.add_argument("-e", dest='epochs', type=int, default=200, help="Number of iterations")
    parser.add_argument("-m", dest='run_mode', default="SubtypeHGAN", help="run_mode: feature, cluster")
    parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-4, help="weight")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument("-p", dest='other_approach', default="spectral", help="kmeans, spectral, tsne_gmm, tsne")
    parser.add_argument("-s", dest='surv_path',
                        default="data/pcawg_donor_clinical_August2016_v9.csv",
                        help="surv input")
    parser.add_argument("-t", dest='type', default="BRCA", help="cancer type: BRCA, GBM")
    args = parser.parse_args()
    print(str(args))

    set_global_determinism()
    model_path = './model/' + args.type + '.h5'
    SubtypeHGAN = SubtypeHGAN_API(model_path, epochs=args.epochs, weight=args.disc_weight)
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'COAD': 4, 'CRC': 4, 'ESCA': 2, 'HNSC': 4, 'KIRC': 4, 'KIRP': 2,
                   'LIHC': 3, 'GBM': 4, 'LGG': 3, 'LUAD': 3, 'LUSC': 4, 'PAAD': 2, 'PCPG': 4, 'PRAD': 3, 'READ': 4,
                   'SARC': 2, 'SKCM': 4, 'STAD': 3, 'THCA': 2, 'UCEC': 4, 'UCS': 2, 'UVM': 4, 'ALL': 28, 'PACA':4}

    if args.run_mode == 'SubtypeHGAN':
        cancer_type = args.type.split('-')[0]
        out_file_apth = '../results/' + cancer_type + '/'
        os.makedirs(out_file_apth,exist_ok=True)

        # if cancer_type not in cancer_dict and args.cluster_num == -1:
        #     print("Please set the number of clusters!")
        # elif args.cluster_num == -1:
        #     args.cluster_num = cancer_dict[cancer_type]

        fea_tmp_file = '../fea/' + cancer_type + '.fea' #
        tmp_dir = '../fea/' + cancer_type + '/'
        os.makedirs(tmp_dir, exist_ok=True)

        ldata = []
        l = []
        nb_line = 0
        for base_file in ["miRNA", 'Mut', 'CNA', 'rna']:
            # # base_file = splitext(basename(line.rstrip()))[0]
            # base_file = line.split('.')[-3].split('/')[-1]
         
            fea_save_file = tmp_dir + base_file + '.fea'   # ./fea/BRCA/rna.fea
            
            if isfile(fea_save_file):
                print("read fea file: ",fea_save_file)
                df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
                l = df_new.index
            else:
                print(f"no {fea_save_file}")
                continue
                clinic_parms = ['# donor_unique_id', 'project_code', 'donor_vital_status', 'donor_survival_time',
                                'donor_interval_of_last_followup', 'donor_sex', 'donor_age_at_diagnosis']
                df = pd.read_csv(args.surv_path, header=0, sep=',',
                                 usecols=clinic_parms)   # 2834
                df = df[df['donor_vital_status'].notnull()]  # (2665, 7)
                df['status'] = np.where(df['donor_vital_status'] == 'deceased', 1, 0)
                df['days'] = df.apply(lambda r: r['donor_survival_time'] if r['donor_vital_status'] == 1 else r[
                    'donor_interval_of_last_followup'],
                                      axis=1)

                df = df[df['days'].notnull()]   # (1757, 9)
                df['acronym'] = df['project_code'].apply(lambda x: str(x).split('-')[0])
                df.index = df['# donor_unique_id']

                if cancer_type == 'ALL':
                    pass
                else:
                    df = df.loc[df['acronym']==cancer_type, ::]
                    if df.shape[0] < 20:
                        print(cancer_type,df.shape)
                        sys.exit(0)
                
                clic_save_file = out_file_apth + cancer_type + '.clinic'

                df_new = pd.read_csv(line.rstrip(), sep=',', header=0, index_col=0)
                nb_line += 1

                if nb_line == 1:   
                    ids = list(df.index)
                    ids_sub = list(df_new)
                    l = list(set(ids) & set(ids_sub))
                    df_clic = df.loc[
                        l, ['status', 'days', 'donor_sex', 'donor_age_at_diagnosis']]
                    df_clic.to_csv(clic_save_file, index=True, header=True, sep=',')
                df_new = df_new.loc[::, l]
                df_new = df_new.fillna(0)
                if 'miRNA' in base_file or 'rna' in base_file:
                    df_new = np.log2(df_new + 1)

                print(f"data type: {base_file}, number: ",df_new.shape)

                scaler = preprocessing.StandardScaler()  # 按照列标准化,index 基因，column 样本
                mat = scaler.fit_transform(df_new.values.astype(float))
                df_new.iloc[::, ::] = mat
                df_new = df_new.T           # index 样本,column基因
                print("最大方差:", df_new.var().max())
                selector = VarianceThreshold(threshold=(0.8))   # 0.8方差过滤,按照列方差(基因）过滤

                try:
                    selector.fit(df_new)
                    df_new = df_new.loc[:, selector.get_support()]
                except:
                    print("error occured!")
                    # log.warning(f"no {base_file} type data")
                    df_new = df_new.iloc[:, [0]]
                    # df_new = pd.DataFrame()
                
                df_new.to_csv(fea_save_file, index=True, header=True, sep=',')
            # df_new = df_new.T
            print(f"cancer type: {args.type}, data type: {base_file}, number: ",df_new.shape)
            ldata.append(df_new.values.astype(float))

        start_time = time.time()
        vec = SubtypeHGAN.feature_gan(ldata, index=l, n_components=100, weight=args.disc_weight)
        df = pd.DataFrame(data=[time.time() - start_time])

        print("save fusioned fea file:",fea_tmp_file, vec.shape)
        vec.to_csv(f"../fea/{cancer_type}/{cancer_type}.SubtypeGAN", header=True, index=True)

        out_file = out_file_apth + cancer_type + '.SubtypeGAN.time'

        print("save run time file: ",df[0][0])
        df.to_csv(out_file, header=True, index=False, sep=',')

        # if isfile(fea_tmp_file):
        #     X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
        #     X['SubtypeHGAN'] = SubtypeHGAN.gmm(args.cluster_num).fit_predict(X.values) + 1
        #     X = X.loc[:, ['SubtypeHGAN']]
        #     out_file = out_file_apth + cancer_type + '.SubtypeGAN'
        #     X.to_csv(out_file, header=True, index=True, sep='\t')
        # else:
        #     print('file does not exist!')

    elif args.run_mode == 'show':
        cancer_type = args.type
        out_file_apth = './results/' + cancer_type + '/'
        fea_tmp_file = './fea/' + cancer_type + '.fea'
        out_file = out_file_apth + cancer_type + '.tsne'
        if isfile(fea_tmp_file):
            df = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            mat = df.values.astype(float)
            labels = SubtypeHGAN.tsne(mat)
            print(labels.shape)
            df['x'] = labels[:, 0]
            df['y'] = labels[:, 1]
            df = df.loc[:, ['x', 'y']]
            df.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'kmeans':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        X['kmeans'] = SubtypeHGAN.kmeans(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['kmeans']]
        out_file = './results/' + cancer_type + '.kmeans'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'spectral':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        X['spectral'] = SubtypeHGAN.spectral(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['spectral']]
        out_file = './results/' + cancer_type + '.spectral'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'hierarchical':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        X['hierarchical'] = SubtypeHGAN.hierarchical(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['hierarchical']]
        out_file = './results/' + cancer_type + '.hierarchical'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'ae':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        fea_save_file = './fea/' + cancer_type + '.ae'
        start_time = time.time()
        vec = SubtypeHGAN.feature_ae(X, n_components=100)
        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_tmp_file, header=True, index=True, sep='\t')
        out_file = './results/' + cancer_type + '.ae.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            X['ae'] = SubtypeHGAN.gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['ae']]
            out_file = './results/' + cancer_type + '.ae'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'vae':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        fea_save_file = './fea/' + cancer_type + '.vae'
        start_time = time.time()
        vec = SubtypeHGAN.feature_vae(X, n_components=100)
        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_tmp_file, header=True, index=True, sep='\t')
        out_file = './results/' + cancer_type + '.vae.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            X['vae'] = SubtypeHGAN.gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['vae']]
            out_file = './results/' + cancer_type + '.vae'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'cc':
        K1_dict = {'BRCA': 4, 'BLCA': 3, 'KIRC': 3,
                   'GBM': 2, 'LUAD': 3, 'PAAD': 2,
                   'SKCM': 3, 'STAD': 3, 'UCEC': 4, 'UVM': 2}
        K2_dict = {'BRCA': 8, 'BLCA': 6, 'KIRC': 6,
                   'GBM': 4, 'LUAD': 6, 'PAAD': 4,
                   'SKCM': 6, 'STAD': 6, 'UCEC': 8, 'UVM': 4}
        # K1_dict = {'ACC': 2, 'BRCA': 4, 'BLCA': 4, 'CRC': 3, 'ESCA': 2, 'HNSC': 3, 'KIRC': 3, 'KIRP': 2,
        #            'LIHC': 2, 'GBM': 3, 'LGG': 2, 'LUAD': 2, 'LUSC': 3, 'PAAD': 2, 'PCPG': 3, 'PRAD': 2, 'SARC': 2,
        #            'SKCM': 3, 'STAD': 2, 'THCA': 2, 'UCEC': 3, 'UCS': 2, 'UVM': 3, 'ALL': 18}
        # K2_dict = {'ACC': 4, 'BRCA': 8, 'BLCA': 8, 'CRC': 6, 'ESCA': 4, 'HNSC': 6, 'KIRC': 6, 'KIRP': 4,
        #            'LIHC': 4, 'GBM': 6, 'LGG': 4, 'LUAD': 4, 'LUSC': 6, 'PAAD': 4, 'PCPG': 6, 'PRAD': 4, 'SARC': 4,
        #            'SKCM': 6, 'STAD': 4, 'THCA': 4, 'UCEC': 6, 'UCS': 4, 'UVM': 6, 'ALL': 30}
        cancer_type = args.type
        base_file = splitext(basename(args.file_input))[0]
        fea_tmp_file = './fea/' + cancer_type + '.fea'
        fs = []
        cc_file = './results/k.cc'
        fp = open(cc_file, 'a')
        if isfile(fea_tmp_file):
            X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            cc = ConsensusCluster(SubtypeHGAN.gmm, K1_dict[cancer_type], K2_dict[cancer_type], 10)
            cc.fit(X.values)
            X['cc'] = SubtypeHGAN.gmm(cc.bestK).fit_predict(X.values) + 1
            X = X.loc[:, ['cc']]
            out_file = './results/' + cancer_type + '.cc'
            X.to_csv(out_file, header=True, index=True, sep='\t')
            fp.write("%s, k=%d\n" % (cancer_type, cc.bestK))
        else:
            print('file does not exist!')
        fp.close()

    elif args.run_mode == 'pca':
        base_file = splitext(basename(args.file_input))[0]
        out_file = './' + base_file + '.pca'
        if isfile(args.file_input):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            mat = X.values.astype(float)
            labels = SubtypeHGAN.pca(mat)
            fea = pd.DataFrame(data=labels, index=X.index,
                               columns=map(lambda x: 'v' + str(x), range(labels.shape[1])))
            print(fea.shape)
            fea.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'preprocess':
        pass


if __name__ == "__main__":
    main()

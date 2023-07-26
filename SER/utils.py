import numpy as np
print("Numpy version: ", np.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import pandas as pd
plt.rcParams['figure.figsize'] = (21,15)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
from tqdm.notebook import tqdm
import parselmouth

import librosa
import librosa.display
import scipy
import random
from sklearn import metrics
from statistics import mean

#import shap
from pycaret.classification import *
from pca import pca
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (21,15)
random.seed(1234)
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



def select_feature_combinations(df, tuning_score, iterations=1000, cols_number=10):
    print('select_feature_combinations')
    pot_cols_list = set()
    all_cols = list(df.columns.unique())
    all_cols.remove('class')
    labels = df['class']
    saved_cols = []
    saved_explained_variances = []
    dic_explained_variances = {}
    
    X_df = df[all_cols]
    Xs_df = StandardScaler().fit_transform(X_df)
    Xcols_df = X_df.columns
    X_df = pd.DataFrame(Xs_df)
    X_df.columns = Xcols_df
    threshold_list = []
    for i in tqdm(range(iterations)):
        if len(all_cols) > cols_number:
            pot_cols = random.sample(all_cols, k=cols_number)
        else:
            pot_cols = all_cols
        if not any(len(set(pot_cols) & set(elem)) == cols_number for elem in pot_cols_list):
            X = df[pot_cols]
            Xs = StandardScaler().fit_transform(X)
            Xcols = X.columns
            X = pd.DataFrame(Xs)
            X.columns = Xcols
            ch_pot_cols = metrics.calinski_harabasz_score(X, labels)
            df_pot_cols = metrics.calinski_harabasz_score(X_df, labels)
            threshold_list.append(ch_pot_cols-df_pot_cols)
            if len([k for k in threshold_list if k>0]) > 0:
                if ch_pot_cols - df_pot_cols > mean([k for k in threshold_list if k>0])+tuning_score:
                    saved_cols.append(pot_cols)
                    pot_cols_list.add(tuple(pot_cols))
    return saved_cols, pot_cols_list


def boosted_dataset_construction(saved_cols, df):
    dic_pca = {}
    for pot_cols in tqdm(saved_cols):
        X = df[pot_cols]
        Xs = StandardScaler().fit_transform(X)
        Xcols = X.columns
        X = pd.DataFrame(Xs)
        X.columns = Xcols
        pca = PCA()
        Z = pca.fit_transform(X)
        Z = pca.inverse_transform(Z)
        dic_pca.update({f'PC_{saved_cols.index(pot_cols)}_1':Z[:,0], 
                        f'PC_{saved_cols.index(pot_cols)}_2':Z[:,1]})
    data_pca = pd.DataFrame(dic_pca)
    data_pca['class'] = list(df['class'])
    return data_pca


def flatten(l):
    return [item for sublist in l for item in sublist]
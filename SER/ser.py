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
#For Google Colab only
#from pycaret.utils import enable_colab 
#enable_colab()
from pca import pca
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (21,15)
random.seed(1234)
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from utils import select_feature_combinations, boosted_dataset_construction, flatten


dataset_name = 'comb'
df = pd.read_csv(f'../../all_handcrafted_data_{dataset_name}.csv')
df = df[df['source'] == 'TESS']
df.drop(columns=['source', 'path'],inplace=True)
df.dropna(inplace=True)
df = df.loc[:,~df.columns.duplicated()].copy()
df = df.replace(np.nan, 0)

df_original = df.loc[:,~df.apply(lambda x: x.duplicated(),axis=1).all()].copy()
most_important_names_list = list(df_original.drop(columns=['class']).columns.unique())

    
flag = True
while flag:
    df = df_original[most_important_names_list+['class']]
    saved_cols, pot_cols_list = select_feature_combinations(df, 0, int(len(most_important_names_list)*3), 20)
    print(len(list(saved_cols)))
    print(df.columns)
    boosted_dataset = boosted_dataset_construction(saved_cols, df)
    print(boosted_dataset.columns)
    X = boosted_dataset.drop(columns=['class'])
    Xs = StandardScaler().fit_transform(X)
    Xcols = X.columns
    X = pd.DataFrame(Xs)
    X.columns = Xcols
    data = X
    data['class']= boosted_dataset['class']
    clf = setup(data=data, target='class', train_size=0.7, session_id=123, data_split_stratify=True, silent=True)
    best_model = compare_models(include=['dt','rf','et','lightgbm','catboost', 'xgboost'])
    best_model_tuned = tune_model(best_model)
    import shap
    explainer = shap.TreeExplainer(best_model_tuned)
    X = boosted_dataset.drop('class', axis=1)
    shap_values = explainer.shap_values(get_config('X_test'))
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns,sum(vals))),columns=['col_name','feature_importance_vals'])
    feature_importance = feature_importance.sort_values(by=['feature_importance_vals'],ascending=False).reset_index()
    most_important_names_list = []
    for i in tqdm(range(len(feature_importance))):
        if feature_importance['feature_importance_vals'][i] > 0:
            comb_index = feature_importance['col_name'][i].split('_')[1]
            corresponding_data_combination = df_original[saved_cols[int(comb_index)]]
            model = PCA(n_components=2).fit(corresponding_data_combination)
            X_pc = model.transform(corresponding_data_combination)
            n_pcs= model.components_.shape[0]
            most_important = [np.argsort(np.abs(model.components_[j]))[-10:] for j in range(n_pcs)]
            initial_feature_names = corresponding_data_combination.columns
            most_important_names = [initial_feature_names[most_important[j]] for j in range(n_pcs)]
            dic = {'PC_{}'.format(j+1): most_important_names[j] for j in range(n_pcs)}
            # build the dataframe
            df_important_features = pd.DataFrame(dic.items())
            most_important_names_list+= most_important_names
    most_important_names_list = list(set(flatten(most_important_names_list)))
    print(best_model_tuned)
    print(most_important_names_list)
    if len(most_important_names_list) <= 20:
        flag = False

evaluate_model(best_model_tuned)
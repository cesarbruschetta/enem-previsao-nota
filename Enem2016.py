#!/usr/bin/env python
# coding: utf-8

# ## CodeNation Challenge: Prediction of the Math Grades of Students in Enem 2016
# Competition Description here: https://www.codenation.com.br/journey/data-science/challenge/enem-2.html

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Import csv's
df_train = pd.read_csv("train.csv")
df_test= pd.read_csv("test.csv")



# Drop Different Columns from train and test
target = 'NU_NOTA_MT'
dif = list(set(df_train.drop(target,axis=1).columns).difference(set(df_test.columns)))
df_train.drop(dif,axis=1,inplace=True)

# Check Columns with constant values
dropcols_train = [c for c in df_train.columns if (df_train[c].nunique()==1) & (df_train[c].isnull().sum() == 0)]

#Columns to be dropped
cols_to_drop = dropcols_train + ['NU_INSCRICAO']


# Store and Drops Id's from datasets, and Target from Train Dataset
ID = 'NU_INSCRICAO'
y_train = df_train[target].values
train_id = df_train[ID].values
test_id = df_test[ID].values


#Merge Data before preprocessing:
df_merge = pd.concat([df_train.drop(target,axis=1),df_test],axis=0)
df_merge.drop(cols_to_drop,axis=1,inplace=True)



#Missing Data on Merge Dataset
total = df_merge.isnull().sum().sort_values(ascending=False)
percent = (df_merge.isnull().sum()/df_merge.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


df_merge['TP_ENSINO'].fillna(4,inplace=True) #'4' will represent the NaN class of this feature
df_merge['TP_DEPENDENCIA_ADM_ESC'].fillna(5,inplace=True) #'5' will represent the NaN class of this feature
df_merge['Q027'].fillna('None',inplace=True) #'None' will represent the NaN class of this feature


print('Missing Data on NU_NOTA_MT:',df_train[target].isnull().sum())
print('Missing Data on NU_NOTA_CN:',df_train['NU_NOTA_CN'].isnull().sum())
print('Missing Data on NU_NOTA_CH:',df_train['NU_NOTA_CH'].isnull().sum())
print('Missing Data on NU_NOTA_REDACAO:',df_train['NU_NOTA_REDACAO'].isnull().sum())


# We can see that the NaN's are equal in NU_NOTA_MT and NU_NOTA_REDACAO, probably because these tests were applied in the same day. Therefore, we are not going to train our model in those days. We will use the feature 'TP_STATUS_REDACAO' to tell which rows we need to drop in df_train and df_test, because the NaN's from this feature indicates the examples which the student missed the Math/Redacao test day.
# Restore datraframes df_train and df_test
df_train = df_merge[:len(df_train)]
df_train[target] = y_train.tolist()
df_test = df_merge[len(df_train):]
df_test[ID] = test_id
df_train[ID] = train_id


# Store the examples in Test dataset which our answer will be NaN based on 'TP_STATUS_REDACAO':
# Store ID's of Test Dataset which we will set the prediction result as NaN
NaNs_ID = df_test.loc[df_test['TP_STATUS_REDACAO'].isnull(),ID]
df_test = df_test[~df_test['TP_STATUS_REDACAO'].isnull()] #Remove those examples from test dataset
df_train = df_train[~df_train['TP_STATUS_REDACAO'].isnull()] #Remove those examples from train dataset


# **Re-merging:** Now we can Re-Merge the datasets into the df_merge DataFrame and end our data-processing steps:
y_train = df_train[target].values
test_id = df_test[ID].values
df_merge = pd.concat([df_train.drop(target,axis=1),df_test],axis=0)
df_merge.drop(ID,axis=1,inplace=True)


# Let's check again the missing data on df_merge
total = df_merge.isnull().sum().sort_values(ascending=False)
percent = (df_merge.isnull().sum()/df_merge.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# So there are still two features with missing values, 'NU_NOTA_CN' and 'NU_NOTA_CH'. These features are the grades of the tests applied in the same day, so they have the exact same number of NaNs. As they are numeric features, we will treat these NaNs with the value -100.
df_merge['NU_NOTA_CN'].fillna(-100,inplace=True)
df_merge['NU_NOTA_CH'].fillna(-100,inplace=True)

print('\nIs there any NaN value  left in the dataset?:',df_merge.isnull().sum().any())

# ### LabelEncoder of the categorical values
qualitative_features = [f for f in df_merge.columns 
                        if (df_merge[f].dtypes == object) | (df_merge[f].dtypes == bool)] #Lista de Features Qualitativas.
quantitative_features = [f for f in df_merge.dropna().columns 
                         if (df_merge[f].dtypes != object) & (df_merge[f].dtypes != bool)] #Lista de Features Qualitativas.

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
d = defaultdict(LabelEncoder)

### Encoding the variable
fit = df_merge[qualitative_features].apply(lambda x: d[x.name].fit_transform(x))
df_merge[qualitative_features] = fit


#### Restore datraframes df_train and df_test
df_train = df_merge[:len(df_train)]
df_train[target] = y_train.tolist()
df_test = df_merge[len(df_train):]


#### Division between X_train,X_val,y_train,y_val

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

size_test = 0.3
df_train = shuffle(df_train) #shuffle data before division
train_target = df_train[target] # Just for code readibility
predictors = df_train.drop(target, axis=1)
X_train, X_val, y_train, y_val = train_test_split(predictors, 
                                                    train_target,
                                                    train_size=1-size_test, 
                                                    test_size=size_test, 
                                                    random_state=0)
X_test = df_test


# ### Create and Train LGBModel
# Custom function to run light gbm model
import lightgbm as lgb
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "n_estimators":1000,
        "num_leaves" : 30,
        "min_child_samples" : 30,
        "learning_rate" : 0.005,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    print('\nTraining LGBM...')
    model = lgb.train(params, lgtrain, 8000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_lgb, lgb_model = run_lgb(X_train, y_train, X_val, y_val, X_test)
y_train_lgb = lgb_model.predict(X_train)
y_val_lgb = lgb_model.predict(X_val)
print('End Training LGBM...')


# ### Feature Importance for LGBM
fig, ax = plt.subplots(figsize=(9,14))
lgb.plot_importance(lgb_model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# ### Build Results DFs for plotting
df_resultado_train = pd.DataFrame(
        {
         'y_train':y_train.astype(float),
         'y_train_lgb':y_train_lgb.astype(float),
                })

df_resultado_val = pd.DataFrame(
        {
         'y_val':y_val.astype(float),  
         'y_val_lgb':y_val_lgb.astype(float),
                })

# Plots
import matplotlib.pyplot as plt
plt.figure(1)
df_resultado_val.sort_values('y_val_lgb',inplace=True)
df_resultado_val.sort_values('y_val',inplace=True)
plt.plot(df_resultado_val['y_val'].tolist(),'r',label='y_val')
plt.plot(df_resultado_val['y_val_lgb'].tolist(),'b-.',label='LGB')
plt.legend()

plt.figure(2)
df_resultado_train.sort_values('y_train_lgb',inplace=True)
df_resultado_train.sort_values('y_train',inplace=True)
plt.plot(df_resultado_train['y_train'].tolist(),'r-',label='y_train')
plt.plot(df_resultado_train['y_train_lgb'].tolist(),'b-.',label='LGB')
plt.legend()


# ### Conclusions and Notes

# Our Model performance in blue is definately not Ideal, it is generalizing speacially badly at low values (close to zero). Maybe an  Outliers analisys could help our model with this particular behaviour. Another ML methods could be compared to LightGB aswell.
# 
# Another topic that could definatelly improve our model is feature engineering.

# ### Submission CSV File Creation

prediction = lgb_model.predict(X_test)
df_saida = pd.DataFrame(
        {'NU_INSCRICAO':list(test_id)+list(NaNs_ID),
         'NU_NOTA_MT':list(prediction)+[np.nan]*len(NaNs_ID)
                })

# ### Submit Solution
df_saida.to_csv('answer.csv',index=False)


######
# This code is copied from https://www.kaggle.com/hyunseokc/detecting-early-alzheimer-s#2.-DATA 
# This source is only used to import the OASIS dataset properly
######

import pandas as pd
import numpy as np
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score

def dementia_loader(USE_COLAB = False, GOOGLE_DRIVE = ''):
    path = 'Data/OAS2_RAW/oasis_longitudinal.csv'
    
    if USE_COLAB:
        path = os.path.join(GOOGLE_DRIVE,path)
        
    df = pd.read_csv(path)
    df.head()

    df = df.loc[df['Visit']==1] # use first visit data only because of the analysis we're doing
    df = df.reset_index(drop=True) # reset index after filtering first visit data
    df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns

    # Check missing values by each column
    pd.isnull(df).sum() 
    # The column, SES has 8 missing values

    # Dropped the 8 rows with missing values in the column, SES
    df_dropna = df.dropna(axis=0, how='any')
    pd.isnull(df_dropna).sum()

    df_dropna['Group'].value_counts()


    # Dataset with deleted rows
    Y = df_dropna['Group'].values # Target for the model
    X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] # Features we use

    X = torch.tensor(X.values).to(device = 'cuda', dtype = torch.float32)
    Y = torch.tensor(Y).to(device = 'cuda', dtype = torch.int32)

    
    # Add column of ones to X:
    temp_ones = torch.ones([X.shape[0], 1]).to(device = X.device, dtype = X.dtype)
    X = torch.cat((X,temp_ones), dim = 1)

    # splitting into three sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=0)

    return x_train, y_train, x_test, y_test
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# Load the data
df = pd.read_csv("./heart_disease_uci.csv")

# Exploratory data analysis
print("number of rows and features : ", df.shape)

# Preprocess the data
# Handle missing values
print("total missing variables : ", df.isnull().sum())
# Deleting the rows with the missing values
df = df.dropna(axis=0)
# Checking if there are any missing variables
print("checking is there are any missiing varibales ..\n", df.isnull().sum())

# Removing the not so important features
del df['ca']
del df['thal']
del df['slope']
del df['oldpeak']

def str_features_to_numeric(data):
    # Transforms all string features of the df to numeric features
    
    # Determination categorical features
    categorical_columns = []
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features = data.columns.values.tolist()
    for col in features:
        if data[col].dtype in numerics: continue
        categorical_columns.append(col)
    
    # Encoding categorical features
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            le.fit(list(data[col].astype(str).values))
            data[col] = le.transform(list(data[col].astype(str).values))
    
    return data

df = str_features_to_numeric(df)
df[['trestbps','chol','thalach']] = df[['trestbps','chol','thalach']].astype(int)

#if it's int64 set it as int32
for column in df.columns:
    if df[column].dtype == 'int64':
        df[column] = df[column].astype('int32')

# Creating the network
model = BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','num'),('fbs','num'),('num','restecg'),('num','thalach'),('num','chol')])

# Fit the model using maximum likelihood estimation
model.fit(df, estimator=MaximumLikelihoodEstimator)


# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from xgboost import XGBClassifier

print('Libraries Imported')


# helper function to load the dataset and separate features from response, and exclude certain columns
def load_dataset(filename, response_col):
    #  load the dataset as a pandas DataFrame
    data =  pd.read_csv(filename ,sep = ',',decimal = ',', encoding = 'unicode_escape', engine ='python')
    # rename the response variable as 'response'
    data = data.rename(columns={response_col:'response'})

    return data


data = load_dataset('dataset_32_pendigits.csv', 'class')
data.head()


#  split into input (X) and output (y) variables
X = data[data.columns.difference(['response'])]
y = data.response


# split into train, test, val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#  Split train into test-val for early training stopping and for checking with small sample of new data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.98, stratify=y_test, random_state=2587)


y_train = np.array(y_train, dtype='int64')
y_val = np.array(y_val, dtype='int64')
print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)


params = { 'eta': 0.1,  # eta between(0.01-0.2) This is the learning rate
            'max_depth': 4,  # values between(3-10)
            'max_delta_step': 0.5,
            'subsample': 0.7,  # values between(0.5-1)
            'colsample_bytree': 0.5,  # values between(0.5-1)
            'tree_method': "auto",
            'lambda': 1,
            'alpha': 0.2,
            'process_type': "default",
            'num_parallel_tree': 1,
            'objective': 'multi:softmax',
            'min_child_weight': 1,
            'booster': 'gbtree',
            'sample_type': "uniform",
            'eval_metric': "merror",
            'num_class': 63}


# Xgb
model = XGBClassifier(params, use_label_encoder=False)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, early_stopping_rounds=10, eval_metric='merror')

# dump model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# dump also original columns in case input data is not complete
with open("columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)


# dump test data as pickle
with open('mydata.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('response.pkl', 'wb') as f:
    pickle.dump(y_test, f)
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# In[5]:


df_train=pd.read_csv("train_mcd.csv")
df_train.shape


# In[13]:


df_train.head()


# In[7]:


y_train=df_train["y"]


# In[8]:


cols = [c for c in df_train.columns if 'X' in c]
print('Number of features: {}'.format(len(cols)))


# In[9]:


print('Feature types:')
df_train[cols].dtypes.value_counts()


# In[10]:


counts = [[], [], []]
for c in cols:
    typ = df_train[c].dtype
    uniq = len(np.unique(df_train[c]))
    if uniq == 1:
        counts[0].append(c)
    elif uniq == 2 and typ == np.int64:
        counts[1].append(c)
    else:
        counts[2].append(c)

print('Constant features: {} Binary features: {} Categorical features: {}\n'
      .format(*[len(c) for c in counts]))
print('Constant features:', counts[0])
print('Categorical features:', counts[2])


# In[25]:


df_test = pd.read_csv('test_mcd.csv')


# In[26]:


usable_columns = list(set(df_train.columns) - set(['ID', 'y']))
y_train = df_train['y'].values
id_test = df_test['ID'].values


# In[27]:


x_train = df_train[usable_columns]
x_test = df_test[usable_columns]


# In[28]:


def check_missing_values(df):
    if df.isnull().any().any():
        print("There are missing values in the dataframe")
    else:
        print("There are no missing values in the dataframe")
check_missing_values(x_train)
check_missing_values(x_test)


# In[29]:


for column in usable_columns:
    cardinality = len(np.unique(x_train[column]))
    if cardinality == 1:
        x_train.drop(column, axis=1) # Column with only one 
        # value is useless so we drop it
        x_test.drop(column, axis=1)
    if cardinality > 2: # Column is categorical
        mapper = lambda x: sum([ord(digit) for digit in x])
        x_train[column] = x_train[column].apply(mapper)
        x_test[column] = x_test[column].apply(mapper)
x_train.head()


# In[30]:


print('Feature types:')
x_train[cols].dtypes.value_counts()


# In[31]:


n_comp = 12
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(x_train)
pca2_results_test = pca.transform(x_test)


# In[32]:


import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_valid, y_train, y_valid = train_test_split(pca2_results_train, y_train, test_size=0.2, random_state=4242)


# In[34]:


d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(pca2_results_test)


# In[35]:


params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4


# In[36]:


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


# In[37]:


watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 
                1000, watchlist, early_stopping_rounds=50, 
                feval=xgb_r2_score, maximize=True, verbose_eval=10)


# In[38]:


p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = p_test
sub.to_csv('Mercedes-Benz-Greener-Manufacturing.csv', index=False)


# In[39]:


sub.head()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# Name : Thakur Dishant Kumar
# 
# 
# 
# 
# 
# Course : BTech
# 
# 
# 
# 
# 
# 
# Branch : CSE
# 
# 
# 
# 
# 
# 
# 
# Univ Roll No: 2018820
# 
# 
# 
# 
# 
# 
# 
# 
# 
# College : Graphic Era Hill University
# 
# 
# 
# 
# 
# 
# 
# 

# ## Import modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv('winequality.csv')
df.head()


# In[3]:


# statistical info
df.describe()


# In[4]:


# datatype info
df.info()


# ## Preprocessing the dataset

# In[5]:


# check for null values
df.isnull().sum()


# In[6]:


# fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())


# In[7]:


df.isnull().sum()


# ## Exploratory Data Analysis

# In[8]:


# create box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col, data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[9]:


# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[10]:


# log transformation
df['free sulfur dioxide'] = np.log(1 + df['free sulfur dioxide'])


# In[11]:


sns.distplot(df['free sulfur dioxide'])


# In[12]:


sns.countplot(df['type'])


# In[13]:


sns.countplot(df['quality'])


# ## Coorelation Matrix

# In[14]:


corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# ## Input Split

# In[15]:


X = df.drop(columns=['type', 'quality'])
y = df['quality']


# ## Class Imbalancement

# In[16]:


y.value_counts()


# In[17]:


# !pip install imbalanced-learn


# In[18]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)


# In[19]:


y.value_counts()


# ## Model Training

# In[20]:


# classify function
from sklearn.model_selection import cross_val_score, train_test_split
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)
    
    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[22]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[24]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model, X, y)


# In[ ]:





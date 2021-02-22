#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing Tools

# ## Importing the libraries

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[31]:


dataset = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\Machine Learning A-Z\All Codes & Files\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[32]:


print(X)


# In[33]:


print(y)


# ## Taking care of missing data

# In[34]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[35]:


print(X)


# ## Encoding categorical data

# ### Encoding the Independent Variable

# In[36]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))


# In[37]:


print(X)


# ### Encoding the Dependent Variable

# In[38]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[39]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[41]:


print(X_train)


# In[42]:


print(X_test)


# In[43]:


print(y_train)


# In[44]:


print(y_test)


# ## Feature Scaling

# In[45]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])


# In[46]:


print(X_train)


# In[47]:


print(X_test)


# In[ ]:





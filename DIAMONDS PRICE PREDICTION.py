#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report , confusion_matrix ,accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # IMPORTING DATASET

# In[2]:


Data = pd.read_csv('C:/CAREER/diamonds.csv')
print(Data)


# # EXPLORATORY DATA ANALYSIS

# In[3]:


Data = Data.drop('Unnamed: 0', axis = 1)


# In[4]:


Data.head(10)


# In[5]:


Data.tail(7)


# In[6]:


Data.shape


# In[7]:


Data.info()


# In[8]:


Data.describe()


# 3. DATA PREPROCESSING
# the following steps are executed in this step
# i) Finding missing values and dropping rows with missing values
# ii) Converting the data to Numpy
# iii) Dividing the data set into training data and test data
# 

# In[9]:


Data.isnull().sum()


# Amazing! There were no missing values in the diamonds dataset
# I will convert the object value columns into integers. These three columns include:cut, color, and clarity

# In[10]:


Data['cut'] = Data['cut'].map({'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1})

Data['color'] = Data['color'].map({'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1})

Data['clarity'] = Data['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})


# In[29]:


Data.describe()


# In[11]:


Data=Data.rename(columns={"x": "Length", "y": "width","z":"height",})
Data


# In[30]:


Data.info()


# In[12]:


x=Data[['carat','cut','color','clarity','depth','price','table','Length','width','height']]
x


# In[13]:


y = Data['price']
print(y)


# 4. DATA VISUALIZATION
# I will python libraries such as Matplotlib and Seaborn to customize appealing plots for effective storytelling

# In[14]:


plt.figure(figsize = (12, 6))
sns.histplot(Data['price'], bins = 20)


# In[15]:


plt.figure(figsize=(5,5))
plt.bar(Data['color'].value_counts().index, Data['color'].value_counts())
plt.ylabel("Number of Diamonds")
plt.xlabel("Color")
plt.show()


# In[18]:


color_pivot = x.pivot_table(index='color', values='price')
color_pivot.plot(kind='bar', color='blue',figsize=(12,6))
plt.xlabel('color')
plt.ylabel('price')
plt.title('Impact of color on price')
plt.xticks(rotation=0)
plt.show()


# In[19]:


plt.figure(figsize = (10, 5))
sns.histplot(Data['carat'], color= "yellow", bins=10)


# In[20]:


plt.figure(figsize=(10, 10))
plt.pie(Data['cut'].value_counts(),labels=['Ideal','Premium','Very Good','Good','Fair'],autopct='%1.1f%%')
plt.title('Cut')
plt.show()


# In[21]:


plt.figure(figsize=(5,5))
plt.bar(x['clarity'].value_counts().index, x['clarity'].value_counts())
plt.title('Clarity')
plt.ylabel("Number of Diamonds")
plt.xlabel("Clarity")
plt.show()


# Grouped Visualization

# In[22]:


color_pivot = x.pivot_table(index='cut', values='price')
color_pivot.plot(kind='bar',figsize=(12,6))
plt.xlabel('cut')
plt.ylabel('price')
plt.title('Impact of cut on price')
plt.xticks(rotation=0)
plt.show()


# In[23]:


plt.figure(figsize = (12, 5))
sns.barplot(x="cut",
            y="price",
            hue="color",
            data=Data)
plt.title("Cut - Price - Color")


# In[24]:


plt.figure(figsize = (12, 5))
sns.barplot(x="cut",
            y="price",
            hue="clarity",
            data = Data)
plt.title("Cut - Price - Clarity")


# In[25]:


sns.jointplot(x = "price", 
              y = Data["carat"], 
              data = Data) 


# In[27]:


Data= sns.load_dataset("diamonds")
sns.pairplot(Data,hue='color')


# In[28]:


sns.displot(Data, x="x", hue='color',kind="kde", fill="True")


# In[34]:


Data['cut'] = Data['cut'].map({'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1})

Data['color'] = Data['color'].map({'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1})

Data['clarity'] = Data['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})


# In[35]:


Data=Data.rename(columns={"x": "Length", "y": "width","z":"height",})
Data


# In[36]:


x = Data.drop('price', axis=1)
y = Data['price']
#print(X_features.head(10))
#data.replace([np.inf, -np.inf], np.nan, inplace=True)
#X_train, X_test, target_train, target_test = train_test_split(X, target, test_size = 0.20)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=40)


# In[37]:


print(x)


# In[38]:


from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(x_train, y_train)
print(DTR)


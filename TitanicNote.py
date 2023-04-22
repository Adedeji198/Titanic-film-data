#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading libraries 

import pandas as pd
import numpy as np

#visual libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().system('pip install seaborn')


# In[3]:


#loading data
titanic = pd.read_csv('titanic.csv')


# In[4]:


titanic.head()


# In[5]:


titanic.columns


# In[6]:


titanic.shape


# In[7]:


titanic.info()


# In[8]:


titanic.isnull().sum()


# In[9]:


titanic.describe()


# # Handling Missing Value

# In[10]:


#dropping irrelevant columns
columns =['PassengerId','Name','Ticket','Cabin']
titanic.drop(columns,axis=1,inplace=True)


# In[11]:


titanic.head()


# In[12]:


titanic.info()


# In[13]:


titanic["Age"].median()


# In[14]:


titanic["Age"].fillna(titanic["Age"].median(),inplace=True)


# In[15]:


titanic.isnull().sum()


# In[16]:


titanic["Embarked"].mode()


# In[17]:


titanic["Embarked"].fillna(titanic["Embarked"].mode()[0],inplace=True)


# In[18]:


titanic.isnull().sum()


# # Exploratory Data Analysis

# In[19]:


titanic["SibSp"].value_counts()


# In[20]:


titanic["Survived"].value_counts()


# In[21]:


titanic["Sex"].value_counts()


# In[22]:


plt.figure(figsize=(4,4))
sns.countplot(x='Survived',data=titanic)


# In[23]:


titanic["Pclass"].value_counts()


# In[24]:


sns.countplot(x='Pclass',data=titanic)


# In[25]:


sns.countplot(x="Pclass",hue="Survived",data=titanic);


# In[26]:



sns.countplot(x="Sex",hue="Survived",data=titanic)


# In[27]:



sns.countplot(x="SibSp",hue="Survived",data=titanic)


# In[28]:



sns.countplot(x="Embarked",hue="Survived",data=titanic)


# In[29]:



sns.countplot(x="Embarked",hue="Pclass",data=titanic)


# In[30]:


titanic["Age"].hist()
plt.title("Age")


# In[31]:


sns.displot(titanic["Age"],kde=True)


# In[32]:


#Correlation

sns.heatmap(titanic.corr(),annot=True)


# # Data Preprocessing

# In[33]:


titanic.info()


# In[34]:


#Feature Encoding
titanic['Embarked'] = titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
titanic['Sex'] = titanic['Sex'].map( {'male': 0, 'female': 1} ).astype(int)


# In[35]:


titanic.info()


# In[36]:


titanic.columns


# In[37]:


X= titanic.drop("Survived",axis=1)
y=titanic['Survived']


# In[38]:


y=titanic['Survived']


# In[39]:


#Splitting Data 
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=42)


# # Model Building

# In[40]:


#Training the Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[41]:


#TRAINNING OUR MODEL


# In[42]:


model.fit(X_train,y_train)


# In[43]:


X_train_prediction=model.predict(X_train)


# In[44]:


#ACCURACY


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


training_data_accuracy=accuracy_score(y_train,X_train_prediction)


# In[47]:


print("Accuracy score of training data:",training_data_accuracy)


# In[48]:


model.fit(X_test,y_test)


# In[49]:


X_test_prediction=model.predict(X_test)


# In[50]:


test_data_accuracy=accuracy_score(y_test,X_test_prediction)


# In[51]:




print("Accuracy score of test data:",test_data_accuracy)


# # Prediction 

# In[52]:


input_data=(3,0,35,0,0,8.05,0)


# In[53]:


input_data_as_numpy_array=np.asarray(input_data)


# In[54]:


#RESHAPED YOUR DATA


# In[55]:


input_data_reshape=input_data_as_numpy_array.reshape(1,-1)


# In[56]:


#PREDICTION USING OUR MODEL


# In[57]:


prediction=model.predict(input_data_reshape)
#print(prediction)
if prediction[0]==0:
    print("NOT SURVIVED")
if prediction[0]==1:
    print("SURVIVED")


# In[ ]:





# In[ ]:





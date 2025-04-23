#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# In[1]:


import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# ### Loading the Dataset

# In[2]:


df = pd.read_csv('Titanic-Dataset.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# ### Checking for any null values or missing values

# In[5]:


df.isnull().sum()


# ### Visualizing the missing values

# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### Visualizing the Survival Count by Passenger Class and Gender

# In[7]:


fig1 = px.bar(df,x = 'Survived',y = 'Sex', orientation='h',color= 'Sex',color_discrete_sequence= px.colors.qualitative.Dark24,
              template = 'simple_white', title= 'Survival Count by Gender')
fig1.update_layout(title_x = 0.5, yaxis_title='')
fig1.update_layout(showlegend = False)
fig1.show()


# In[8]:


fig2 = px.bar(df,x = 'Survived',y = 'Pclass', orientation='h',color= 'Pclass',color_discrete_sequence= px.colors.qualitative.Dark24,
              template = 'simple_white', title= 'Survival Count by Passenger Class')
fig2.update_layout(title_x = 0.5, yaxis_title='')
fig2.update_layout(showlegend = True)
fig2.show()


# In[9]:


fig3 = px.bar(df,x = 'Survived',y = 'Pclass',color= 'Sex',barmode='group',orientation='h',color_discrete_sequence= px.colors.qualitative.Dark24,
              template = 'simple_white', title= 'Survival Count by Passenger Class and Gender')
fig3.update_layout(title_x = 0.5, yaxis_title='')
fig3.update_layout(showlegend = True)
fig3.show()


# ### Check for Duplicate values

# In[10]:


df.duplicated().sum()


# ### Statistical Measurement

# In[11]:


df.describe(include= 'all')


# ### Filling the missing values

# #### For Age, the missing values are filled by the median value of the age per passenger class

# In[12]:


df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))


# #### For Embarked, the missing values are filled by mode value

# In[13]:


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# ### Droping the unwanted columns and prepare the data to train the model

# In[14]:


df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df


# ### Converting the categorical columns into numerical to perform the Logistic Regression

# In[15]:


df = pd.get_dummies(df, columns =['Sex'], drop_first=True)
df.head(2)


# In[16]:


df = pd.get_dummies(df, columns=['Embarked'])
df.head(2)


# ### Defining features and target

# In[17]:


X = df.drop('Survived', axis=1)
y = df['Survived']
print(X)
print(y)


# ### Spliting the dataset into train and test data

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Fitting the data into the Logistic Regression Model

# In[19]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ### Prediction of Survivals and Evaluating the Model

# In[20]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", matrix)
print("Report:\n",classification_report(y_test, y_pred))


# In[21]:


print(classification_report(y_test, y_pred))


# In[22]:


X.head(5)


# In[23]:


y.head(5)


# ## Testing the Model with some sample dataset

# In[24]:


X_Sample =pd.DataFrame([{'Pclass':1,'Age':25.0,'SibSp':0,'Parch':0,'Fare':71.2833,'Sex_male':True,'Embarked_C':False,'Embarked_Q':True,'Embarked_S':False},
                          {'Pclass':3,'Age':32.0,'SibSp':1,'Parch':2,'Fare':26.00,'Sex_male':False,'Embarked_C':True,'Embarked_Q':False,'Embarked_S':False},
                          {'Pclass':2,'Age':39.0,'SibSp':0,'Parch':0,'Fare':53.00,'Sex_male':False,'Embarked_C':False,'Embarked_Q':False,'Embarked_S':True}])
X_Sample


# In[25]:


y1_Pred = model.predict(X_Sample)
print('The predicted survivals for Sample_data:\n', y1_Pred)


# In[ ]:





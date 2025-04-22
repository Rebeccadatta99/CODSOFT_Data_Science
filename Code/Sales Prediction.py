#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction

# ## Importing the libraries

# In[63]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ## Loading the Dataset

# In[15]:


df = pd.read_csv('Advertising.csv')
df.head(5)


# ## Analyzing the Dataset

# In[16]:


df.shape


# In[17]:


df.info()


# ## Checking for null and duplicate values

# In[18]:


df.isnull().sum()


# In[19]:


df.duplicated().sum()


# ## Statistical measurement

# In[24]:


df.describe()


# ## Visualizing the dataset and checking the Correlation

# In[27]:


TV_Sales = df[['TV','Sales']]
TV_Sales


# In[28]:


fig1 = px.scatter(df,x = 'TV', y = 'Sales', 
                 template = 'simple_white', title = 'Sales by TV advertising')
fig1.update_layout(title_x = 0.5, xaxis_title = 'Advertising Expenses',yaxis_title='Sales')
fig1.show()
print('The correlation coefficient of TV and Sales:\n',TV_Sales.corr())


# In[29]:


Radio_Sales = df[['Radio','Sales']]
Radio_Sales


# In[30]:


fig2 = px.scatter(df,x = 'Radio', y = 'Sales', 
                 template = 'simple_white', title = 'Sales by Radio advertising')
fig2.update_layout(title_x = 0.5, xaxis_title = 'Advertising Expenses',yaxis_title='Sales')
fig2.show()
print('The correlation coefficient of Radio and Sales:\n',Radio_Sales.corr())


# In[32]:


Newspaper_Sales = df[['Newspaper','Sales']]
Newspaper_Sales


# In[33]:


fig3 = px.scatter(df,x = 'Newspaper', y = 'Sales', 
                 template = 'simple_white', title = 'Sales by Newspaper advertising')
fig3.update_layout(title_x = 0.5, xaxis_title = 'Advertising Expenses',yaxis_title='Sales')
fig3.show()
print('The correlation coefficient of Newspaper and Sales:\n',Newspaper_Sales.corr())


# In[34]:


fig4 = px.scatter(df,x = ['TV','Radio','Newspaper'], y = 'Sales', 
                 template = 'simple_white', title = 'Sales by advertising')
fig4.update_layout(title_x = 0.5, xaxis_title = 'Advertising Expenses',yaxis_title='Sales')
fig4.show()
print('The correlation coefficient:\n',df.corr())


# ## Defining Features and Target

# In[56]:


X = df.drop(['Sales'],axis=1)
X.shape
X.head(4)


# In[57]:


y = df[['Sales']]
y.shape
y.head(4)


# ## Spliting the Dataset into train and test data

# In[58]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# ## Fitting the data into the Linear Regression model

# In[59]:


model = LinearRegression()
model.fit(X_train,y_train)


# ## Prediction of Sales

# In[60]:


y_pred = model.predict(X_test)
y_pred


# ## Evaluating the Model

# In[66]:


mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ",mse)
r2 = r2_score(y_test, y_pred)
print("r2 score: ",r2)


# In[62]:


print(df['Sales'].min(), df['Sales'].max(), df['Sales'].mean())


# In[67]:


Sales_range = df['Sales'].max()-df['Sales'].min()
print('Sales range: ',Sales_range)


# In[68]:


rmse = np.sqrt(mse)
print('Root mean square error: ',rmse)


# #### Sales range is 1.6 to 27, so the total spread is 25.4.
# #### The model’s error (1.55) is about 6% of the total range, which is quite low. 
# #### The model also have the r2_score of 0.91 which express that the model is quite strong.

# ## Testing the model with some sample dataset

# In[69]:


X.head()


# In[70]:


y.head()


# In[71]:


X_sample = pd.DataFrame([{'TV': 140.2, 'Radio': 42.5, 'Newspaper': 50.5}])
X_sample


# In[72]:


y1_pred = model.predict(X_sample)
y1_pred


# In[ ]:





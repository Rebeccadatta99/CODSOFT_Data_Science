#!/usr/bin/env python
# coding: utf-8

# # Movie Rating Prediction

# ## Importing the Libraries

# In[1]:


import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ## Loading the Dataset

# In[2]:


df = pd.read_csv('IMDb Movies India.csv',encoding='ISO-8859-1',low_memory=False)
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# ## Checking for null values or missing values

# In[5]:


df.isnull().sum()


# ## Checking for duplicate values in the dataset

# In[6]:


df.duplicated().sum()


# In[7]:


df = df.drop_duplicates()  #droping the duplicate values


# In[8]:


df = df.dropna(subset=['Rating'])  #droping the null rows based on the rating column


# In[9]:


df.isnull().sum()


# ## Visualizing the null values

# In[10]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Statistical Measurement

# In[11]:


df.describe(include='all')


# ## Extracting the numerical part from the Duration column and Store it into a new column

# In[12]:


df['Duration(min)'] = df['Duration'].str.extract(r'(\d+)')
df['Duration(min)'] = pd.to_numeric(df['Duration(min)'], errors='coerce')
df.head(1)


# ## Converting the Year column into numerical column

# In[13]:


df['Year']=df['Year'].str.replace('(', '').str.replace(')', '').astype('int64')


# ## Checking the data types of the columns

# In[14]:


df.info()


# ## Filling the null values

# In[15]:


df['Duration(min)'].fillna(df['Duration(min)'].median(),inplace=True)


# In[16]:


df['Genre'] = df['Genre'].fillna('')


# In[17]:


for col in ['Director','Actor 1','Actor 2','Actor 3']:
    df[col]= df[col].fillna('Unknown')


# In[18]:


df['Votes']= df['Votes'].str.replace(",","").astype('int64')


# ## Spliting the Genre column as it has more than one value in a single cell

# In[19]:


df['Genre'] = df['Genre'].str.replace(' ', '')
genre_dummies = df['Genre'].str.get_dummies(sep=',')
genre_dummies


# In[20]:


genre_dummies.columns = genre_dummies.columns.str.strip()


# In[21]:


df = pd.concat([df,genre_dummies],axis=1)


# In[22]:


print(df.shape)


# ## Droping the unwanted columns and prepare the Dataset to train the model

# In[23]:


df.drop(['Name','Genre','Duration'],axis=1,inplace=True)


# In[24]:


df['Director_encoded'] = df.groupby('Director')['Rating'].transform('mean')
df['Actor 1_encoded'] = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor 2_encoded'] = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor 3_encoded'] = df.groupby('Actor 3')['Rating'].transform('mean')


# ## Distribution of IMDb Ratings

# In[25]:


fig = px.histogram(df, x='Rating', nbins=20, title='Distribution of IMDb Ratings',
                   template = 'simple_white')
fig.update_layout(title_x = 0.5,xaxis_title='Ratings', yaxis_title='')
fig.show()


# ## Votes vs Ratings

# In[26]:


fig1 = px.scatter(df, x = 'Votes', y = 'Rating', title= 'Votes vs IMDb Rating', template = 'simple_white')
fig1.update_layout(title_x = 0.5, xaxis_title= 'Votes', yaxis_title = 'Ratings')
fig1.show()


# In[27]:


Correlation = df[['Votes','Rating']]
print("The correlation coefficient between votes and ratings:\n",Correlation.corr())


# ## Top 10 Genre by Movie Count

# In[28]:


genre_cols = [col for col in df.columns if col not in ['Year', 'Rating','Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3',
                                                       'Duration(min)', 'Director_encoded', 'Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded' ]]
genre_counts = df[genre_cols].sum().sort_values(ascending=False)
genre_df = genre_counts.reset_index()
genre_df.columns = ['Genre', 'Movie Count']


# In[29]:


genre_df.head(10)


# In[29]:


fig2 = px.bar(genre_df.head(10), x = 'Movie Count', y = 'Genre', orientation = 'h',
              title = 'Top 10 Genre by movie count', template='simple_white')
fig2.update_layout(title_x = 0.5, showlegend = False)
fig2.show()


# ## Movies Released Per Year (Trend)

# In[30]:


yearly_counts = df['Year'].value_counts().sort_index()
fig3 = px.line(x=yearly_counts.index, y=yearly_counts.values, title='Number of Movies Released Per Year', template = 'simple_white')
fig3.update_layout(title_x = 0.5, xaxis_title= 'Year', yaxis_title = 'No. of Movies released')
fig3.show()


# ## Defining Features and Target

# In[31]:


X = df.drop(['Rating','Director','Actor 1','Actor 2','Actor 3'],axis=1)
X.head(5)


# In[32]:


y = df['Rating']
y.head()


# ## Spliting the dataset into Train and Test data

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Fitting the data into Linear Regression Model

# In[34]:


model = LinearRegression()
model.fit(X_train, y_train)


# ## Prediction of Movie ratings

# In[35]:


y_pred = model.predict(X_test)
y_pred


# ## Evaluating the Model

# In[36]:


mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:\n",mse)
r2 = r2_score(y_test, y_pred)
print("r2 score:\n",r2)


# In[37]:


fig4 = px.scatter(df,x = y_test, y = y_pred, 
                 template = 'simple_white', title = 'Actual vs Predicted Ratings')
fig4.update_layout(title_x = 0.5, xaxis_title = 'Actual Rating',yaxis_title='Predicted Rating')
fig4.show()


# ## Testing the model with some sample data

# In[38]:


pd.set_option('display.max_columns', None)


# In[39]:


X.head(1)


# In[40]:


X_sample = pd.DataFrame([{'Year':2020,'Votes':20,'Duration(min)':120.0,'Action':1,'Adventure':1,'Animation':0,'Biography':0,'Comedy':0,
                          'Crime':0,'Documentary':0,'Drama':0,'Family':0,'Fantasy':0,'History':0,'Horror':0,'Music':0,'Musical':0,'Mystery':0,
                          'News':0,'Romance':0,'Sci-Fi':0,'Sport':0,'Thriller':0,'War':0,'Western':0,
                          'Director_encoded':7.0,'Actor 1_encoded':6.5,'Actor 2_encoded':7.5,'Actor 3_encoded':7.0}])
X_sample


# In[41]:


y1_pred = model.predict(X_sample)
y1_pred


# In[ ]:





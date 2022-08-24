#!/usr/bin/env python
# coding: utf-8

# ## Ascent Real Estate - Price Predictor 
# 

# In[1]:


import pandas as pd 


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing ['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50, figsize=(20, 15))


# ## Train-Test Splitting 

# In[10]:


import numpy as np
np.random.seed(42)
def split_train_test(data,test_ratio):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


print(f"Rows in train set : {len(train_set)} \n Rows in test set : { len(test_set)} \n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set : {len(train_set)} \n Rows in test set : { len(test_set)} \n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index , test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


#95/7


# In[18]:


#376/28


# In[19]:


housing = strat_train_set.copy()


# ## Looking for Correlations

# In[20]:


corr_matric = housing.corr()
corr_matric['MEDV'].sort_values(ascending = False)


# In[21]:


corr_matric['MEDV'].sort_values(ascending = False)


# In[22]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV" , "RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize = (12,8))


# In[23]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# ## Trying Out Attribute Combinations 
# 

# In[24]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[25]:


housing.head()


# In[26]:


corr_matric = housing.corr()
corr_matric['MEDV'].sort_values(ascending = False)


# In[27]:


housing.shape


# In[28]:


housing.describe() #before we started filling missing attributes 


# In[29]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[30]:


housing = strat_train_set.drop("MEDV",axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# In[31]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[32]:


imputer.statistics_


# In[33]:


x=imputer.transform(housing)


# In[42]:


housing_tr = pd.DataFrame(x,columns = housing.columns)


# In[43]:


housing_tr.describe()


#  ## Scikit - learn Design

# #Primarily ,  three types of objects 
# 1.Estimators - It estimates some parameter based on a dataset . eg imputer 
# It has a fit method and  transform method 
# fit method - fits the dataset and calculations internal parameters 
# 
# 2.Transformers  - it takes input and returns output based on the learnings from fit(). it also has a covenience function called fit_transfrom() which fits and then transforms 
# 
# 3.predictors - Linear regression model is an example of predictor . fir() and predict()  are two common functions . it also gives score fucntion which will evaluate the predictions.

# ## Features Scaling 

# In[44]:


primarily two types of methods : 
       1. Min-max [normalization] : (value -min / max-min )
           sklearn provides a class called MinMaxScaler for this 
       2.standardization = 
       (value - mean)/std 
       sklearn provides a class called standard scaler for this 
       


# ## Creating Pipelining 

# In[45]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")) ,
    ('std_scaler'  ,StandardScaler()),
])


# In[46]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[47]:


housing_num_tr.shape


# ## selecting a desired model for ascent real estate 

# In[72]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
#model=LinearRegression()
#model=DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[73]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[74]:


prepared_data = my_pipeline.transform(some_data)


# In[75]:


model.predict(prepared_data)


# In[76]:


list(some_labels)


# ## Evauating the model 

# In[77]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[78]:


rmse


# ## Using Better Evaluation - Cross Validation 

# In[79]:


from sklearn.model_selection import cross_val_score 
scores = cross_val_score (model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[80]:


rmse_scores


# In[82]:


def print_scores(scores):
    print ("Scores :" , scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())


# In[83]:


print_scores(rmse_scores)


# ## Saving the model
# 

# In[84]:


from joblib import dump,load
dump(model,'Ascent.joblib')


# ## testing the model 

# In[87]:


X_test = strat_test_set.drop("MEDV",axis=1)


# In[93]:


Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)


# In[89]:


final_rmse


# In[94]:


prepared_data[0]


# In[ ]:





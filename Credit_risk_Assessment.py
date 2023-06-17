#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Credit Risk Assessment
 # Import necessary libraries

#------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loading the Data
credit_data=pd.read_csv(r'C:/Users/MAYANK/Downloads/credit_data/credit_data.csv')

credit_data= credit_data.dropna()
credit_data = credit_data[credit_data['age'] >= 0]
credit_data=credit_data.drop("clientid", axis=1)
credit_data['age'] = credit_data['age'].astype(int)
#Splitting the data into features(X) and target variable (Y)
X=credit_data.drop('default',axis=1)
y=credit_data['default']

#pre-processing the data


#Visual representation
import matplotlib.pyplot as plt
X.hist(bins=50, figsize=(20,15))
plt.show()



# In[38]:


attributes = ["income", "age", "loan"]

# Create the scatter matrix
pd.plotting.scatter_matrix(X[attributes], figsize=(12, 8))

# Show the scatter matrix
plt.show()


# In[39]:


correlation_matrix = X.corr()
print(correlation_matrix)


# In[40]:


#heat map
import seaborn as sns

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[41]:


#split the data into training an dtesting sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#perform feture scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
# Train a logistic regression model
model=LogisticRegression()
model.fit(X_train_scaled,y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:





# In[ ]:





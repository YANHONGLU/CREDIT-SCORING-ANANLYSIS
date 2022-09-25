#!/usr/bin/env python
# coding: utf-8

# # Step 1 Import Libraries

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# # Step 2 Importing dataset

# In[70]:


dataset = pd.read_csv(r'C:\Users\Yanhong\Desktop\python_projects\scoredcard_etl.csv')


# # Step 3 Building Model

# In[25]:


# Train Test Split


# In[71]:


y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values


# In[72]:


# splitting dataset into training and test (in ratio 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[73]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[74]:


# Exporting Normalisation Coefficients for later use in prediction
import joblib
joblib.dump(sc, r'C:\Users\Yanhong\Desktop\python_projects\scoredcard_normalisation')


# In[30]:


# Risk Model building-Train and fit a logistic regression model on the training set.


# In[75]:


classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[76]:


# Exporting Logistic Regression Classifier for later use in prediction
import joblib
joblib.dump(classifier, r'C:\Users\Yanhong\Desktop\python_projects\classifier_scorecard')


# In[77]:


## Evaluation


# In[78]:


print(confusion_matrix(y_test,y_pred))


# In[79]:


print(accuracy_score(y_test, y_pred))


# # Step 4 Writing output file

# In[80]:


predictions = classifier.predict_proba(X_test)
predictions


# In[83]:


# writing model output file
df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])
dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_csv(r"C:\Users\Yanhong\Desktop\python_projects\scoredcard_model_prediction.csv", sep=',', encoding='UTF-8')
dfx.head()


# In[ ]:





# In[ ]:





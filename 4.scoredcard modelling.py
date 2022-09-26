#!/usr/bin/env python
# coding: utf-8
# # Step 1 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# # Step 2 Importing dataset
dataset = pd.read_csv(r'C:\Users\Yanhong\Desktop\python_projects\scoredcard_etl.csv')

# # Step 3 Building Model
# Train Test Split
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values
# splitting dataset into training and test (in ratio 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Exporting Normalisation Coefficients for later use in prediction
import joblib
joblib.dump(sc, r'C:\Users\Yanhong\Desktop\python_projects\scoredcard_normalisation')

# Risk Model building-Train and fit a logistic regression model on the training set.
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Exporting Logistic Regression Classifier for later use in prediction
import joblib
joblib.dump(classifier, r'C:\Users\Yanhong\Desktop\python_projects\classifier_scorecard')

## Evaluation
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# # Step 4 Writing output file
predictions = classifier.predict_proba(X_test)
predictions
# writing model output file
df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])
dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_csv(r"C:\Users\Yanhong\Desktop\python_projects\scoredcard_model_prediction.csv", sep=',', encoding='UTF-8')
dfx.head()

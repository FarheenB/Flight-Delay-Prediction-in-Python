#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold  


# In[2]:


#get flights dataset
flights_data = pd.read_csv('dataset/FlightDelays.csv')


# In[3]:


#converting the object type(String type) to categorical data
#substitute to adding dummy variables

from collections import defaultdict
d = defaultdict(LabelEncoder)

#selecting cols that need to be transformed
df = pd.DataFrame(flights_data, columns = ['CARRIER', 'DEST', 'FL_DATE', 'ORIGIN','TAIL_NUM','Flight_Status'])

# Encoding the variable
fit = df.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
flights_df=pd.DataFrame(df.apply(lambda x: d[x.name].transform(x)))

#add the rest of the cols to the dataframe
flights_df['CRS_DEP_TIME']=flights_data['CRS_DEP_TIME']
flights_df['DEP_TIME']=flights_data['DEP_TIME']
flights_df['DISTANCE']=flights_data['DISTANCE']
flights_df['FL_NUM']=flights_data['FL_NUM']
flights_df['Weather']=flights_data['Weather']
flights_df['DAY_WEEK']=flights_data['DAY_WEEK']
flights_df['DAY_OF_MONTH']=flights_data['DAY_OF_MONTH']

#print top 5 values of the dataset
flights_df.head()


# In[4]:


#select dependent and independent variables

X = flights_df.drop({'Flight_Status'}, axis=1)
y = flights_df['Flight_Status']


# # Variable selection and reduction in the size of the model

# In[5]:


# Create VarianceThreshold object with a variance with a threshold of 0.5
thresholder = VarianceThreshold(threshold=1.5)

# Conduct variance thresholding
X_high_variance =pd.DataFrame(thresholder.fit_transform(X))


# In[6]:


#high variance features
X_high_variance.head()


# In[7]:


#Variable Selection- droping the less useful features
X_filter = flights_df.drop({'Weather','DEST','ORIGIN','Flight_Status'}, axis=1)
#reduced model
X_filter.head()


# In[8]:


#split dataset to training and test data 60:40 ratio
X_train, X_test, y_train, y_test = train_test_split(X_filter, y, test_size=0.4, random_state=12)


# # Comparing the data models that fits the best

# # Logistic Regression

# In[9]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logreg = classifier.predict(X_test)


# In[10]:


from sklearn.metrics import accuracy_score
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_logreg)))


# # Decision Tree Classifier

# In[11]:


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dtc = classifier.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_dtc)))


# # Random Forest Classifier

# In[13]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rand = classifier.predict(X_test)


# In[14]:


# Model Accuracy, how often is the classifier correct?
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_rand)))


# # K-NN Classifier

# In[15]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)


# In[16]:


# Model Accuracy, how often is the classifier correct?
print('Accuracy of KNN classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_knn)))


# # SVM Classifier

# In[17]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svc = classifier.predict(X_test)


# In[18]:


# Model Accuracy, how often is the classifier correct?
print('Accuracy of Support vector classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_svc)))


# # Naive Bayes 

# In[19]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_naive = classifier.predict(X_test)


# In[20]:


# Model Accuracy, how often is the classifier correct?
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred_naive)))


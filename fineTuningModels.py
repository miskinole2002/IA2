import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from descriptor import GLCM1, bitdesc
import os
import joblib
import streamlit as st 

# Linear Algorithms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Nonlinear Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# model
models = [('LDA', LinearDiscriminantAnalysis()), 
          ('KNN', KNeighborsClassifier(n_neighbors=10)),
          ('Naive Bayes', GaussianNB()),
          ('Decision Tree', DecisionTreeClassifier()),
          ('SVM', SVC(C=2.5, max_iter=5000)),
          ('Random Forest', RandomForestClassifier()),
          ('AdaBoost', AdaBoostClassifier())
          ]
# Data Transformation
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, Binarizer
# transforms = [('NoTransform', None),
#               ('Rescale', MinMaxScaler()), 
#               ('Normalization', Normalizer()),
#               ('Standardization', StandardScaler())
#               ]
transforms=[('Normalization', Normalizer())]

# model =RandomForestClassifier(random_state=40)
model= AdaBoostClassifier()

metrics = accuracy_score

# Load Signatures / Feature vector
load_signatures = np.load('signatures_GLCM1.npy')
# Split inputs / outputs
X = load_signatures[ : , : -1].astype('float')
Y = load_signatures[ : , -1].astype('int')
# Define test proportion
train_proportion = 0.15
seed = 10
# Split train / test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_proportion, random_state=seed)
# Add transforms
for trans_name, trans in transforms:
   
    st.write(trans_name)

    scaler = trans
    (X_tr, X_te) = (X_train, X_test) if trans_name == 'NoTransform' else (scaler.fit_transform(X_train), scaler.fit_transform(X_test))
    # Train the model
   
  
        
      
       
    
    model.fit(X_tr, Y_train)
                # Evaluation
    Y_pred = model.predict(X_te)
    # st.write(Y_pred)          
    result = metrics(Y_pred, Y_test)
             
    result = metrics(Y_pred, Y_test)
    st.write(result)

# import joblib
# joblib.dump(model,'AdaBoost.joblib')
joblib.dump(scaler,'Scale_AdaBoost.joblib')


# import cv2
# width = 256
# height = 256
# img=cv2.imread('./iris1.jpg')

# img=cv2.resize(img,(width, height))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# feat =  GLCM1(img)
# new_data = np.array([feat])
# scale=StandardScaler()
# new_data = scaler.transform(new_data)
   
              
# pred = model.predict(new_data)
# proba = model.predict_proba(new_data)
# st.write('prediction',pred)
# st.write('probabilite',proba)          
               

#class_list={'fire':1,'nofire':2,'iris-setosa':3,'iris-versicolour':4,'iris-virginica':5}


                    






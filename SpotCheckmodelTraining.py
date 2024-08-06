import numpy as np
from descriptor import GLCM1
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
# Linear Algorithms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Nonlinear Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os
# Data Transformation
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, Binarizer
transforms = [('NoTransform', None),
              ('Rescale', MinMaxScaler()), 
              ('Normalization', Normalizer()),
              ('Standardization', StandardScaler())
              ]
#('Binarization', Binarizer(threshold=0.0))

models = [('LDA', LinearDiscriminantAnalysis()), 
          ('KNN', KNeighborsClassifier(n_neighbors=10)),
          ('Naive Bayes', GaussianNB()),
          ('Decision Tree', DecisionTreeClassifier()),
          ('SVM', SVC(C=2.5, max_iter=5000)),
          ('Random Forest', RandomForestClassifier()),
          ('AdaBoost', AdaBoostClassifier())
          ]

metrics = [('Accuracy', accuracy_score)]

# Features Loading 
#featureLoadGlcm=np.load('./signatures_GLCM1.npy') 

featureLoadBit=np.load('./signatures_bitdesc2.npy')

featureLoadGlcm=np.load('./signatures_GLCM1.npy') 
# st.write(featureLoadBit)
#st.write(featureLoadGlcm)

#st.write(np.load('./iris_glcm.npy'))
signature=[featureLoadBit,featureLoadGlcm]
# st.write(np.load('./signatures_GLCM1.npy'))
st.write(np.load('./signatures_bitdesc2.npy'))

X= featureLoadBit[:,: -2]
#st.write(X)

Y=featureLoadBit[:, -2].astype('int')
#st.write(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y ,test_size=0.15,random_state=10)
#st.write(train_test_split(X,Y ,test_size=0.15,random_state=10))
for trans_name, trans in transforms:
    st.write(trans_name)
    scaler= trans
    (X_tr,X_te)=(X_train,X_test)if trans_name=='NoTransform' else(scaler.fit_transform(X_train), scaler.fit_transform(X_test))
    for metr_name, metric in metrics:
        # st.write(metr_name,'*')

        for  mod_name, model in models:
                classifier = model
                classifier.fit(X_tr, Y_train)
               # st.write(classifier.fit(X_tr, Y_train))
                # Evaluation
                Y_pred = classifier.predict(X_te)
                # st.write('pediction',Y_pred)
                if metr_name == 'Accuracy':
                    result = metric(Y_pred, Y_test)
                else:
                    result = metric(Y_pred, Y_test, average='macro')

                st.write(mod_name,metr_name,'*',result*100)


                #prediction
              
            #     import cv2
            #     img=cv2.imread('./iris1.jpg')

            #     img=cv2.resize(img,(256, 256))
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     feat =  GLCM1(img)
            #     new_data = np.array([feat])
            #    # st.write(metr_name,'*')
                 
            #    ## st.write(classifier)
            #     model=RandomForestClassifier()
            #     pred = RandomForestClassifier.predict(new_data)
            #     proba = RandomForestClassifier.predict_proba(new_data)
            #     st.write('prediction',pred)
            #     st.write('probabilite',proba)
               
    
                    



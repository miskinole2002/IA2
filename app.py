from descriptor import GLCM,bitdesc
from PIL import Image
import numpy as np
from distances import manhattan,canbera,chebyshev,eucludienne,retrieve_similar_image,retrieve_similar_image2
path,patha,pathb='image\img.jpg','image\img2.jpg','image\img3.jpg'
import streamlit as st
import joblib
import page as pg

def main():
    st.title("moteur de recherche ")
    files=['jpeg','jpg','png']

    on = st.toggle("Activate feature")

    if on:
        st.write("Feature activated!")
        uploaded_files = st.file_uploader('telecharger', files) 
   
        if uploaded_files:
            st.image(uploaded_files, caption="Image téléchargée", use_column_width=False)
        
            with open("uploaded_image.png", "wb") as f:
                    f.write(uploaded_files.getbuffer())
        
                    st.success("Image enregistrée avec succès !")

            featGLCM= GLCM('./uploaded_image.png')
            featBit=bitdesc('./uploaded_image.png')
            #st.write(featureGLCM)
            #st.write(featureBit)
           


    
    
    st.sidebar.header("menu des options")
    radio_options=['GLCM','BiT']
    radio_value=st.sidebar.radio('Descriptor',radio_options)
   # st.write(radio_value)
    dropdown_option=['manhattan','eucludienne','chebyshev','canbera']
    dropdown_value=st.sidebar.selectbox('Distance',dropdown_option)
    slider_value=st.sidebar.slider('nombre image',0,100,50)
    #
    radio_option=['KNN','Naive Bayes','Decision Tree','SVM','Random Forest','AdaBoost']
    radio_value1=st.sidebar.radio('Models',radio_option)
# #load 

    featureGlcm=np.load('./signatures_GLCM4.npy')
    featureBit=np.load('./signatures_bitdesc2.npy')
    # st.write(featureBit)

 # load model
    if(radio_value1=='KNN'):
        model=joblib.load('./models/KNN.joblib')  
        scaler=joblib.load('./Scaler/standizer_KNN.joblib')  
    elif(radio_value1=='Naive Bayes'):
        model=joblib.load('./models/Naive_Bayes.joblib')  
        # scaler=joblib.load('./Scaler/scale_Naive.joblib') 
        scaler=False 
    elif(radio_value1=='Decision Tree'):
        model=joblib.load('./models/Decision_Tree.joblib')  
        # scaler=joblib.load('./Scaler/scale_Decision.joblib') 
        scaler=False 

    elif(radio_value1=='SVM'):
        model=joblib.load('./models/SVM.joblib')  
        scaler=joblib.load('./Scaler/scale_Svm.joblib')
    elif(radio_value1=='Random Forest'):
        model=joblib.load('./models/random_forest.joblib')  
        scaler=joblib.load('./Scaler/standizer.joblib')  
    elif(radio_value1=='AdaBoost'):
        model=joblib.load('./models/AdaBoost.joblib')  
        # scaler=joblib.load('./Scaler/Scale_AdaBoost.joblib') 
        scaler=False 
         
    # st.write(model,scaler)
# #comparaison
    
    if(radio_value=='BiT'):
         sign=np.load('./signatures_bitdesc.npy') #featureBit
         qf= bitdesc('./uploaded_image.png') #featBit
    elif( radio_value=='GLCM'):
        sign= np.load('./signatures_GLCM4.npy') # featureGlcm
        qf=GLCM('./uploaded_image.png') #featGLCM
        

   
    Page1,Page2,page3=st.tabs(['CBIR','CBIR++','CBIR++'])
   
    
    query_features=GLCM('./uploaded_image.png') if (radio_value=='GLCM') else(bitdesc('./uploaded_image.png'))
    sc=np.array([ query_features])
    if scaler:
        sc=scaler.transform(sc)
    else:
        sc=sc
    mo=model.predict(sc)
    features_db=sign
    distance=dropdown_value
    num_results=slider_value


    results1=  retrieve_similar_image(features_db, query_features, distance, num_results,str(mo[0]))
    
    # st.write(results)
    with Page1:
        results=  retrieve_similar_image2(features_db, query_features, distance, num_results)
        # st.write(results)
        for result in results:
            # st.image(result[0])
            #if(result[2]==str(mo[0])):

                st.image(result[0])
                st.write(result[2])
            
    with Page2:
        results1=  retrieve_similar_image(features_db, query_features, distance, num_results,str(mo[0]))
        # st.write(results1)
        for result in results1:
            # st.image(result[0])
            #if(result[2]==str(mo[0])):

                st.image(result[0])
                st.write(result[3])  

if __name__=='__main__' :
    main()
    
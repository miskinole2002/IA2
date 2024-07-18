from descriptor import GLCM,bitdesc
from PIL import Image
import numpy as np
from distances import manhattan,canbera,chebyshev,eucludienne,retrieve_similar_image
path,patha,pathb='image\img.jpg','image\img2.jpg','image\img3.jpg'
import streamlit as st

def main():
    st.title("moteur de recherche ")
    files=['jpeg','jpg','png']
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
    st.write(radio_value)
    dropdown_option=['manhattan','eucludienne','chebyshev','canbera']
    dropdown_value=st.sidebar.selectbox('Distance',dropdown_option)
    slider_value=st.sidebar.slider('nombre image',0,100,50)

#load 
    featureGlcm=np.load('./signatures_GLCM.npy')
    featureBit=np.load('./signatures_bitdesc.npy')
    #st.write(featGlcm)
    #st.write(featureBit)

    
#comparaison

    if(radio_value=='BiT' ):
         sign=featureBit
         qf=featBit
    elif( radio_value=='GLCM' ):
        sign=featureGlcm
        qf=featGLCM

         

    query_features=qf
    features_db=sign
    distance=dropdown_value
    num_results=slider_value
    results=  retrieve_similar_image(features_db, query_features, distance, num_results)
    #st.write(results)
    for result in results:
         st.image(result[0])
         st.write(result[2])
         
   # st.write(results[0][0])
   # st.image(results[0][0])

if __name__=='__main__' :
    main()
    
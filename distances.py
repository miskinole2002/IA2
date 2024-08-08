import numpy as np
from scipy.spatial import distance
import streamlit as st

#manhattan on covertir les liste en array 
def manhattan(v1,v2): #v1 et v2 doivent toujours etre de meme taille
    v1,v2=np.array(v1).astype('float'),np.array(v2).astype('float')
    
    dist=np.sum(np.abs(v1-v2))
    return dist

#eucludienne
def eucludienne(v1,v2): #v1 et v2 doivent toujours etre de meme taille
    v1,v2=np.array(v1).astype('float'),np.array(v2).astype('float')
    
    dist= np.sqrt(np.sum(v1-v2)**2)
    return dist

#chebyshev
def chebyshev(v1,v2): #v1 et v2 doivent toujours etre de meme taille
    v1,v2=np.array(v1).astype('float'),np.array(v2).astype('float')
    
    dist=np.max(np.abs(v1-v2))
    return dist

#canbera

def  canbera(v1,v2):
    return distance.canberra(v1,v2)



def retrieve_similar_image(features_db, query_features, distance, num_results,mod):
    distances = []
    for instance in features_db:
        features, label, img_path,name = instance[ : -3], instance[-3], instance[-2],instance[-1]
        # st.write(mod)
        # st.write(label)
        if label==mod:
            if distance == 'manhattan':
                dist = manhattan(query_features, features)
            if distance == 'eucludienne':
                dist = eucludienne(query_features, features)
            if distance == 'chebyshev':
                dist = chebyshev(query_features, features)
            if distance == 'canbera':
                dist = canbera(query_features, features)
            distances.append((img_path, dist, label,name))
    distances.sort(key=lambda x: x[1])
    return distances[ : num_results]

def retrieve_similar_image2(features_db, query_features, distance, num_results):
    distances = []
    for instance in features_db:
            features, label, img_path = instance[ : -3], instance[-1], instance[-2]
        # st.write(mod)
        # st.write(label)
    
            if distance == 'manhattan':
                dist = manhattan(query_features, features)
            if distance == 'eucludienne':
                dist = eucludienne(query_features, features)
            if distance == 'chebyshev':
                dist = chebyshev(query_features, features)
            if distance == 'canbera':
                dist = canbera(query_features, features)
            distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[ : num_results]

def retrieve_with_image_name(features_db, num_results,mod):
    distances = []
    for instance in features_db:
            features, label, img_path,name = instance[ : -3], instance[-3], instance[-2],instance[-1]
            # st.write(label ,mod)
            if label== mod:
                # st.write(img_path,label)
                distances.append((img_path,  label,name))

            distances.sort(key=lambda x: x[1])
    # st.write(distances)
    return distances[ : num_results]
import numpy as np
from scipy.spatial import distance

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
from skimage.feature import graycomatrix,graycoprops
from BiT import bio_taxo
import cv2
import numpy as np

# fonction glcm

def GLCM(image_path):
    
    Image=cv2.imread(image_path,0) #quand on met 0 il convertira l image en noir sur blanc automatiquement 
    #creation de la matrice de co_occurence
    co_matrix=graycomatrix(Image,[1],[0,np.pi/4],None,symmetric=False,normed=False) 
    # sur le level c est mieux de mettre None pourqu'il considere toutes les valeur de pixel soit en 8 bits ou 6 bits
    # calcul des valeurs 

    cont=graycoprops(co_matrix,"contrast")[0,0]
    dissimilarity=graycoprops(co_matrix,"dissimilarity") [0,0]

    hom=graycoprops(co_matrix,'homogeneity',)[0,0]
    asm=graycoprops(co_matrix,'ASM')[0,0]
    ener=graycoprops(co_matrix,'energy')[0,0]

    corr=graycoprops(co_matrix,'correlation')[0,0]
  

    return[cont,dissimilarity,hom,asm,ener,corr]

def bitdesc(image_path):
    Image=cv2.imread(image_path,0)
    return bio_taxo(Image)
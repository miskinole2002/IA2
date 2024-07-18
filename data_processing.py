
import cv2,os, numpy as np
from descriptor import GLCM,bitdesc
from PIL import Image


def extract_features(image_path,descriptor):
    Img=cv2.imread(image_path)

    if Img is not None:
        features=descriptor(Img)
        return features
    else:
        pass
descriptors=[GLCM,bitdesc]
def procces_datasets(root_folder):
    all_features_GLCM=[]
    all_features_bit=[]

    
        #parcoure tous les dossiers retourne une root,files(liste de tous les images)
    for root,dirs,files in os.walk(root_folder): 
        # print("r=,",root)
        # print("dir=",dirs)
        # print("f:",files)
        for file in files:
           # print(file)
            if file.lower().endswith(('.jpeg','.png','jpg')):
                path_base=os.path.join(root,file)
                #relative_path=os.path.relpath(path_base,root_folder)#permet de construire un chemin
                #file_name=f'{relative_path.split("/")[0]}_{file}'
                folder_name=os.path.basename(os.path.dirname(path_base))
                #features=extract_features(relative_path,GLCM)

                #GLCM

                # features_GLCM=GLCM(path_base)
                # features_GLCM=features_GLCM+[folder_name,path_base]
                # all_features_GLCM.append(features_GLCM)
               
              
                #bitdesc
                try:
                    features_bitdesc=bitdesc(path_base)
                    features_bitdesc=features_bitdesc+[folder_name,path_base]
                    all_features_bit.append(features_bitdesc)
                except Exception as e:
                       print(e)

    # signatures_GLCM=np.array(all_features_GLCM)
    # np.save('signatures_GLCM.npy',signatures_GLCM)
        #bitdesc
    signatures_Bitdesc=np.array(all_features_bit)
    np.save('signatures_bitdesc.npy',signatures_Bitdesc)


    print ('bien enregistre ')

procces_datasets('.\datasets')
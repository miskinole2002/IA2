from IA2.descriptor import GLCM,bitdesc
from IA2.distances import manhattan,canbera,chebyshev,eucludienne
path,patha,pathb='image\img.jpg','image\img2.jpg','image\img3.jpg'

def main():
    
    feat_glcma,feat_glcm2,feat_glcm3=GLCM(path),GLCM(patha),GLCM(pathb)
    distancepath_path=f''' mahan {manhattan(feat_glcm2,feat_glcma)} | {manhattan(feat_glcm2,feat_glcm3)}
                           canbe {canbera(feat_glcm2,feat_glcma)} | {canbera(feat_glcm2,feat_glcm3)}
                           cheb {chebyshev(feat_glcm2,feat_glcma)} | {chebyshev(feat_glcm2,feat_glcm3)}
                           eucl {eucludienne(feat_glcma,feat_glcm2)} | {eucludienne(feat_glcma,feat_glcm3)}


'''
    
    print(distancepath_path)
    
    

if __name__=='__main__' :
    main()
    
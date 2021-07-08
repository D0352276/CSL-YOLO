from .bboxes_trans import BoxesAugment
import numpy as np
import random
import cv2
from tools import ResizeImg,TransformBboxes

def Resize(img,bboxes,target_hw):
    h,w=np.shape(img)[:2]
    w_rate=target_hw[1]/w
    h_rate=target_hw[0]/h
    for i,bbox in enumerate(bboxes):
        x,y,w,h,wht,label=bbox
        x=float(x*w_rate)
        y=float(y*h_rate)
        w=float(w*w_rate)
        h=float(h*h_rate)
        wht=float(wht)
        bboxes[i]=[x,y,w,h,wht,label]
    img=cv2.resize(img,(target_hw[1],target_hw[0]))
    return img,bboxes 

def Mixup(img1,bboxes1,img2,bboxes2):
    if(random.random()>0.5):
        lam=np.random.beta(1.5,1.5)
        img=lam*img1+(1-lam)*img2
        bboxes1=list(map(lambda x:[*x[:4],lam*x[4],x[5]],bboxes1))
        bboxes2=list(map(lambda x:[*x[:4],(1-lam)*x[4],x[5]],bboxes2))
        bboxes=bboxes1+bboxes2
    else:
        img=img1
        bboxes=bboxes1
    return img,bboxes

def ImgAugment(img,bboxes):
    img,bboxes=BoxesAugment(img,bboxes)
    return img,bboxes
    
def MixupAugment(img1,bboxes1,img2,bboxes2,target_hw=[320,320]):
    img1,bboxes1=ImgAugment(img1,bboxes1)
    img2,bboxes2=ImgAugment(img2,bboxes2)
    img1,bboxes1=Resize(img1,bboxes1,target_hw)
    img2,bboxes2=Resize(img2,bboxes2,target_hw)
    img,bboxes=Mixup(img1,bboxes1,img2,bboxes2)
    return img,bboxes
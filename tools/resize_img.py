import numpy as np
import cv2

def ResizeImg(img,out_size,normalize=False):
    img_h,img_w=np.shape(img)[:2]
    if(img_h>=img_w):
        scale=out_size/img_h
        resized_h=out_size
        resized_w=int(img_w*scale)
    else:
        scale=out_size/img_w
        resized_h=int(img_h*scale)
        resized_w=out_size

    img=cv2.resize(img,(resized_w,resized_h))
    img=img.astype(np.float32)
    if(normalize==True):
        img=img/255.
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        img=img-mean
        img=img/std
    pad_h=out_size-resized_h
    pad_w=out_size-resized_w
    img=np.pad(img,[(0,pad_h),(0,pad_w),(0,0)],mode='constant')
    return img,scale

def TransformBboxes(bboxes,scale):
    for i in range(len(bboxes)):
        bboxes[i][0]*=scale
        bboxes[i][1]*=scale
        bboxes[i][2]*=scale
        bboxes[i][3]*=scale
    return bboxes

def DeTransformBboxes(bboxes,scale):
    for i in range(len(bboxes)):
        bboxes[i][0]/=scale
        bboxes[i][1]/=scale
        bboxes[i][2]/=scale
        bboxes[i][3]/=scale
    return bboxes
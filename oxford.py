import scipy.io
import cv2
import numpy as np
import os
from tools import Bboxes2JSON

def Corner2Bbox(corner,label):
    corner=np.array(corner)
    corner_y=corner[...,0:1]
    corner_x=corner[...,1:2]

    minx=np.min(corner_x)
    maxx=np.max(corner_x)
    miny=np.min(corner_y)
    maxy=np.max(corner_y)
    bbox=[minx,miny,maxx-minx,maxy-miny,label]
    return bbox

def Mat2Bboxes(mat_path):
    annotation=scipy.io.loadmat(mat_path)
    bboxes=annotation["boxes"][0]
    out_bboxes=[]
    for i,bbox in enumerate(bboxes):
        bbox=bbox[0][0]
        pos1=bbox[0][0]
        pos2=bbox[1][0]
        pos3=bbox[2][0]
        pos4=bbox[3][0]
        try:label=str(bbox[4][0]).lower()+"_hand"
        except:label="hand"
        corner=[pos1,pos2,pos3,pos4]
        bbox=Corner2Bbox(corner,label)
        out_bboxes.append(bbox)
    return out_bboxes

def MatDir2JsonDir(mat_dir,out_dir):
    mats=os.listdir(mat_dir)
    for i,mat in enumerate(mats):
        try:main_name,sub_name=mat.split(".")
        except:continue
        if(sub_name!="mat"):continue
        print(i)
        mat_path=mat_dir+"/"+mat
        save_path=out_dir+"/"+main_name+".json"
        bboxes=Mat2Bboxes(mat_path)
        Bboxes2JSON(bboxes,save_path)
    return 

MatDir2JsonDir("dataset/oxford_hand/annotations","dataset/oxford_hand/json")


import numpy as np
import cv2
import random

def FlipImg(img,bboxes):
    img_w=np.shape(img)[1]
    img=cv2.flip(img,1)
    for i,bbox in enumerate(bboxes):
        bbox[0]=img_w-bbox[0]-bbox[2]
        bboxes[i]=bbox
    return img,bboxes
def CropImg(img,bboxes):
    h,w=np.shape(img)[:2]
    min_x=min(list(map(lambda x:x[0],bboxes)))
    max_x=max(list(map(lambda x:x[0]+x[2]-1,bboxes)))
    min_y=min(list(map(lambda x:x[1],bboxes)))
    max_y=max(list(map(lambda x:x[1]+x[3]-1,bboxes)))
    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y

    crop_xmin=max(0,int(min_x-random.uniform(0,max_l_trans)))
    crop_ymin=max(0,int(min_y-random.uniform(0,max_u_trans)))
    crop_xmax=max(w,int(max_x+random.uniform(0,max_r_trans)))
    crop_ymax=max(h,int(max_y+random.uniform(0,max_d_trans)))

    img=img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
    bboxes=list(map(lambda x:[float((x[0]-crop_xmin)),
                              float((x[1]-crop_ymin)),
                              float(x[2]),
                              float(x[3]),
                              float(x[4]),
                              x[5]],bboxes))
    return img,bboxes
def AffineImg(img,bboxes):
    h,w=np.shape(img)[:2]
    min_x=min(list(map(lambda x:x[0],bboxes)))
    max_x=max(list(map(lambda x:x[0]+x[2]-1,bboxes)))
    min_y=min(list(map(lambda x:x[1],bboxes)))
    max_y=max(list(map(lambda x:x[1]+x[3]-1,bboxes)))
    max_l_trans=min_x
    max_u_trans=min_y
    max_r_trans=w-max_x
    max_d_trans=h-max_y

    tx=random.uniform(-(max_l_trans-1),(max_r_trans-1))
    ty=random.uniform(-(max_u_trans-1),(max_d_trans-1))

    M=np.array([[1,0,tx],[0,1,ty]])
    img=cv2.warpAffine(img,M,(w,h))

    bboxes=list(map(lambda x:[float(x[0]+tx),float(x[1]+ty),float(x[2]),float(x[3]),float(x[4]),x[5]],bboxes))
    return img,bboxes

def RotateImg(img,bboxes,border_value=(128,128,128)):
    rotate_degree=np.random.uniform(low=-10,high=10)
    h,w=img.shape[:2]
    # Compute the rotation matrix.
    M=cv2.getRotationMatrix2D(center=(w/2,h/2),
                              angle=rotate_degree,
                              scale=1)
    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle=np.abs(M[0, 0])
    abs_sin_angle=np.abs(M[0, 1])
    # Compute the new bounding dimensions of the image.
    new_w=int(h*abs_sin_angle+w*abs_cos_angle)
    new_h=int(h*abs_cos_angle+w*abs_sin_angle)
    # Adjust the rotation matrix to take into account the translation.
    M[0,2]+=new_w//2-w//2
    M[1,2]+=new_h//2-h//2 
    # Rotate the image.
    img=cv2.warpAffine(img,M=M,dsize=(new_w, new_h),flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=border_value)
    new_bboxes=[]
    for bbox in bboxes:
        x,y,w,h,wht,label=bbox
        x1,y1,x2,y2=[x,y,x+w,y+h]
        points=M.dot([[x1,x2,x1,x2],[y1,y2,y2,y1],[1,1,1,1]])
        # Extract the min and max corners again.
        min_xy=np.sort(points,axis=1)[:,:2]
        min_x=np.mean(min_xy[0])
        min_y=np.mean(min_xy[1])
        max_xy=np.sort(points,axis=1)[:,2:]
        max_x=np.mean(max_xy[0])
        max_y=np.mean(max_xy[1])
        bbox=[float(min_x),float(min_y),float(max_x-min_x),float(max_y-min_y),float(wht),label]
        new_bboxes.append(bbox)
    return img,new_bboxes

def BboxShift(img_hw,bbox,shift_range=0.1):
    x,y,w,h,wht,label=bbox
    w_shift_range_b=int(w*shift_range*(-0.5))
    w_shift_range_e=w_shift_range_b*(-1)
    h_shift_range_b=int(h*shift_range*(-0.5))
    h_shift_range_e=h_shift_range_b*(-1)
    x=x+random.randint(w_shift_range_b,w_shift_range_e)
    y=y+random.randint(h_shift_range_b,h_shift_range_e)
    w=w+random.randint(w_shift_range_b,w_shift_range_e)
    h=h+random.randint(h_shift_range_b,h_shift_range_e)
    if(x<0):x=0
    if(y<0):y=0
    if(x+w-1>=img_hw[1]):w=img_hw[1]-x-1
    if(y+h-1>=img_hw[0]):h=img_hw[0]-y-1
    bbox=[float(x),float(y),float(w),float(h),float(wht),label]
    return bbox

def BboxesShift(img,bboxes,shift_range=0.1):
    img_hw=np.shape(img)[:2]
    bboxes=list(map(lambda x:BboxShift(img_hw,x,shift_range),bboxes))
    return img,bboxes

def BoxesAugment(img,bboxes):
    if(random.random()>0.5):
        img,bboxes=FlipImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=BboxesShift(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=CropImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=AffineImg(img,bboxes)
    if(random.random()>0.5):
        img,bboxes=RotateImg(img,bboxes)
    return img,bboxes
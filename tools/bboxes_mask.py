import numpy as np

def GridsMask(anchors,mask_hw):
    anchors=np.array(anchors)
    anchors_len=np.shape(anchors)[0]
    mask_h,mask_w=mask_hw
    h_idx=np.arange(mask_h)
    h_idx=np.expand_dims(h_idx,axis=0)
    h_idx=np.tile(h_idx,[mask_w,1])
    h_idx=h_idx.transpose()
    w_idx=np.arange(mask_w)
    w_idx=np.expand_dims(w_idx,axis=0)
    w_idx=np.tile(w_idx,[mask_h,1])
    xy_mask=np.stack([w_idx,h_idx],axis=-1)
    xy_mask=np.expand_dims(xy_mask,axis=2)
    xy_mask=np.tile(xy_mask,[1,1,anchors_len,1])
    anchors=np.reshape(anchors,[1,1,anchors_len,2])
    anchors=np.tile(anchors,[mask_h,mask_w,1,1])
    grids_mask=np.concatenate([xy_mask,anchors],axis=-1)
    return grids_mask

def BboxesMask(bboxes,mask_hw):
    bboxes=np.array(bboxes)
    bboxes_len,elemts_len=np.shape(bboxes)
    mask_h,mask_w=mask_hw
    bboxes=np.reshape(bboxes,[1,1,bboxes_len,elemts_len])
    bboxes_mask=np.tile(bboxes,[mask_h,mask_w,1,1])
    return bboxes_mask

def NormalizedBboxesMask(bboxes,mask_hw):
    bboxes=np.array(bboxes)
    bboxes_len,elemts_len=np.shape(bboxes)
    mask_h,mask_w=mask_hw
    bboxes=np.reshape(bboxes,[1,1,bboxes_len,elemts_len])
    bboxes_mask=np.tile(bboxes,[mask_h,mask_w,1,1])
    bboxes_xywh=bboxes_mask[...,:4]/np.array([mask_w,mask_h,mask_w,mask_h])
    bboxes_mask=np.concatenate([bboxes_xywh,bboxes_mask[...,4:]],axis=-1)
    return bboxes_mask

def OffsetMask(grids_mask,bboxes_mask):
    grids_mask=np.expand_dims(grids_mask,3)
    bboxes_mask=np.expand_dims(bboxes_mask,axis=2)
    other_elemts=np.tile(bboxes_mask[...,4:],[1,1,np.shape(grids_mask)[2],1,1])
    
    xy_offset=bboxes_mask[...,:2]-grids_mask[...,:2]
    wh_offset=np.log(bboxes_mask[...,2:4]/grids_mask[...,2:4])
    offset_mask=np.concatenate([xy_offset,wh_offset,other_elemts],axis=-1)
    return offset_mask

def IoUMask(grids_mask,bboxes_mask,mask_hw):
    mask_h,mask_w=mask_hw
    grids_mask=np.expand_dims(grids_mask,3)
    bboxes_mask=np.expand_dims(bboxes_mask,axis=2)
    
    grids_xy=grids_mask[...,0:2]
    grids_wh=grids_mask[...,2:4]
    grids_wh_half=grids_wh/2.
    grids_mins=grids_xy-grids_wh_half
    grids_maxes=grids_xy+grids_wh_half
    avaliable_girds=(grids_mins>=0.0).astype(np.float32)+ \
                    (grids_maxes[...,0:1]<mask_w).astype(np.float32)+ \
                    (grids_maxes[...,1:2]<mask_h).astype(np.float32)
    avaliable_girds=np.sum(avaliable_girds,axis=-1)
    avaliable_girds=(avaliable_girds==6.0).astype(np.float32)

    gt_xy=bboxes_mask[...,0:2]
    gt_wh=bboxes_mask[...,2:4]
    gt_wh_half=gt_wh/2.
    gt_mins=gt_xy-gt_wh_half
    gt_maxes=gt_xy+gt_wh_half

    intersect_mins=np.maximum(grids_mins,gt_mins)
    intersect_maxes=np.minimum(grids_maxes,gt_maxes)
    intersect_wh=np.maximum(intersect_maxes-intersect_mins,np.zeros_like(intersect_maxes))
    intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
    grids_areas=grids_wh[...,0]*grids_wh[...,1]
    gt_areas=gt_wh[...,0]*gt_wh[...,1]
    union_area=grids_areas+gt_areas-intersect_area
    iou=intersect_area/union_area
    iou_mask=iou*avaliable_girds
    return iou_mask

def PostiveMask(iou_mask,iou_thres=0.5):
    best_iou=np.max(iou_mask,axis=-1)
    best_iou=np.expand_dims(best_iou,axis=-1)
    pstv_mask_1=(iou_mask==best_iou).astype(np.float32)
    pstv_mask_2=(np.sum(pstv_mask_1,axis=-1)==1.0).astype(np.float32)
    pstv_mask_2=np.expand_dims(pstv_mask_2,axis=-1)
    pstv_mask_3=(iou_mask>=iou_thres).astype(np.float32)
    pstv_mask=((pstv_mask_1+pstv_mask_2+pstv_mask_3)==3.0).astype(np.float32)
    pstv_mask=np.expand_dims(pstv_mask,axis=-1)
    return pstv_mask


# bboxes=[[24,31,30,40,0.14,0,0,0,1],[42,14,23,14,0.24,0,0,1,0],[10,25,24,47,0.34,0,0,0,1]]
# anchors=[[5,5],[10,10],[20,40],[40,40]]
# mask_hw=[64,64]


# bboxes_mask=BboxesMask(bboxes,mask_hw)
# norm_bboxes_mask=NormalizedBboxesMask(bboxes,mask_hw)
# bboxes_mask=np.concatenate([bboxes_mask[...,:4],norm_bboxes_mask[...,:4],bboxes_mask[...,4:]],axis=-1)

# grids_mask=GridsMask(anchors,mask_hw)
# offset_mask=OffsetMask(grids_mask,bboxes_mask)
# iou_mask=IoUMask(grids_mask,bboxes_mask,mask_hw)
# pstv_mask=PostiveMask(iou_mask)
# offset_mask=offset_mask*pstv_mask
# offset_mask=np.sum(offset_mask,axis=-2)



# print(np.shape(offset_mask))
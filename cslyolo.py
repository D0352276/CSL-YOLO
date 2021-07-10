import tensorflow as tf
from modules import ConvBN,InputBIFusion,CSLModule,CSLFPN,VanillaFPN
import math

class CSLConv(tf.Module):
    def __init__(self,anchors_list,labels_len,name="cslconv"):
        super(CSLConv,self).__init__(name=name)
        self._anchors_list=anchors_list
        self._anchors_num=len(self._anchors_list[0])
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._outconv=ConvBN(self._anchors_num*(5+self._labels_len+1),
                             kernel_size=(1,1),
                             use_bn=False,
                             activation=None,
                             name=self._name+"_outconv")
    @tf.Module.with_name_scope
    def _Grids(self,featmap_hw):
        featuremap_hight_idx=tf.range(start=0,limit=featmap_hw[0])
        featuremap_hight_idx=tf.expand_dims(featuremap_hight_idx,axis=0)
        featuremap_hight_idx=tf.tile(featuremap_hight_idx,[featmap_hw[1],1])
        featuremap_hight_idx=tf.transpose(featuremap_hight_idx)

        featuremap_width_idx=tf.range(start=0,limit=featmap_hw[1])
        featuremap_width_idx=tf.expand_dims(featuremap_width_idx,axis=0)
        featuremap_width_idx=tf.tile(featuremap_width_idx,[featmap_hw[0],1])

        grids=tf.stack([featuremap_width_idx,featuremap_hight_idx],axis=-1)
        grids=tf.reshape(grids,[1,featmap_hw[0],featmap_hw[1],1,2])
        grids=tf.cast(grids,tf.float32)
        return grids
    @tf.Module.with_name_scope
    def _RestructInTensor(self,input_ts,anchors):
        ftmp_hw=input_ts.get_shape().as_list()[1:3]
        ftmp_wh=tf.cast(tf.reverse(ftmp_hw,[-1]),tf.float32)

        feature_map=self._outconv(input_ts)
        feature_map=tf.reshape(feature_map,[-1,ftmp_hw[0],ftmp_hw[1],self._anchors_num,5+self._labels_len+1])

        box_for_fit=tf.concat([tf.sigmoid(feature_map[...,0:2]),feature_map[...,2:4]],axis=-1)
        pred_xy=(tf.sigmoid(feature_map[...,0:2])+self._Grids(ftmp_hw))/ftmp_wh

        pred_wh=(anchors+feature_map[...,2:4])/ftmp_wh
        pred_wh=pred_wh*tf.cast(pred_wh>=0,tf.float32)
        ones_mask=tf.cast(pred_wh<=1,tf.float32)
        pred_wh=pred_wh*ones_mask+(1-ones_mask)
        
        pred_box=tf.concat([pred_xy,pred_wh],axis=-1)
        pred_cnfd=tf.sigmoid(feature_map[...,4:5])
        pred_classes=tf.sigmoid(feature_map[...,5:])
        output_ts=tf.concat([box_for_fit,pred_box,pred_cnfd,pred_classes],axis=-1)
        return output_ts
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        output_ts_list=[]
        for i,anchors in enumerate(self._anchors_list):
            x=self._RestructInTensor(input_ts_list[i],anchors)
            output_ts_list.append(tf.identity(x))
        return output_ts_list

class CSLHead(tf.Module):
    def __init__(self,
                 orig_img_hw,
                 labels_len,
                 max_boxes_per_cls=10,
                 score_threshold=0.5,
                 iou_threshold=0.5,
                 nms_type="category_nms",
                 name="cslhead"):
        super(CSLHead,self).__init__(name=name)
        self._orig_img_hw=orig_img_hw
        self._labels_len=labels_len
        self._max_boxes_per_cls=max_boxes_per_cls
        self._score_threshold=score_threshold
        self._iou_threshold=iou_threshold
        self._nms_type=nms_type
        self._name=name
    @tf.Module.with_name_scope
    def _PreprocessPredTs(self,pred_tensor):
        pred_box=pred_tensor[...,4:8]
        pred_cnfd=pred_tensor[...,8:9]
        pred_classes=pred_tensor[...,9:]

        pred_scores=pred_cnfd*pred_classes
        pred_scores=tf.reduce_max(pred_scores,axis=-1)
        pred_mask=pred_scores>=self._score_threshold
        pred_box=tf.boolean_mask(pred_box,pred_mask)

        pred_xy=pred_box[...,0:2]
        pred_wh=pred_box[...,2:4]
        pred_x1y1=pred_xy-(pred_wh/2.)
        pred_x2y2=pred_x1y1+pred_wh
        pred_box=tf.concat([pred_x1y1,pred_x2y2],axis=-1)

        pred_scores=tf.boolean_mask(pred_scores,pred_mask)
        pred_scores=tf.expand_dims(pred_scores,axis=-1)
        pred_classes=tf.boolean_mask(pred_classes,pred_mask)
        pred_classes=tf.cast(tf.reshape(tf.argmax(pred_classes,axis=-1),[-1,1]),tf.float32)
        pred_tensor=tf.concat([pred_box,pred_scores,pred_classes],axis=-1)

        pred_x=pred_xy[...,0]*self._orig_img_hw[1]
        pred_y=pred_xy[...,1]*self._orig_img_hw[0]
        pred_w=pred_wh[...,0]*self._orig_img_hw[1]
        pred_h=pred_wh[...,1]*self._orig_img_hw[0]
        mask_1=tf.cast(pred_x<0,tf.float32)+tf.cast(pred_y<0,tf.float32)
        mask_2=tf.cast(pred_x>=self._orig_img_hw[1],tf.float32)+tf.cast(pred_y>=self._orig_img_hw[0],tf.float32)
        mask_3=tf.cast(pred_w<3,tf.float32)+tf.cast(pred_h<3,tf.float32)
        mask_4=tf.cast(pred_w>=self._orig_img_hw[1],tf.float32)+tf.cast(pred_h>=self._orig_img_hw[0],tf.float32)
        pred_mask=(mask_1+mask_2+mask_3+mask_4)==0
        pred_tensor=tf.boolean_mask(pred_tensor,pred_mask)
        return pred_tensor
    @tf.Module.with_name_scope
    def _PostProcessPredTs(self,pred_tensor):
        pred_box=pred_tensor[...,:4]
        pred_scores=pred_tensor[...,4:5]
        pred_classes=pred_tensor[...,5:6]

        pred_x1y1=pred_box[...,0:2]
        pred_x2y2=pred_box[...,2:4]
        pred_wh=pred_x2y2-pred_x1y1
        pred_xy=pred_x1y1
        pred_xy=tf.clip_by_value(pred_xy,0.0,1.0-1e-8)
        pred_wh=tf.clip_by_value(pred_wh,1e-8,1.0-pred_xy)

        pred_box=tf.concat([pred_xy,pred_wh],axis=-1)
        pred_box=pred_box*tf.cast(tf.stack([self._orig_img_hw[1],
                                            self._orig_img_hw[0],
                                            self._orig_img_hw[1],
                                            self._orig_img_hw[0]],axis=0),tf.float32)


        _,top_indices=tf.nn.top_k(tf.squeeze(pred_scores,axis=-1),k=tf.minimum(100,tf.shape(pred_scores)[0]))
        pred_box=tf.gather(pred_box,top_indices)
        pred_scores=tf.gather(pred_scores,top_indices)
        pred_classes=tf.gather(pred_classes,top_indices)

        pred_tensor=tf.concat([pred_box,pred_scores,pred_classes],axis=-1)
        return pred_tensor
    @tf.Module.with_name_scope
    def _NMS(self,pred_tensor,max_bboxes=100,iou_thres=0.5,score_thres=0.01):
        pred_box=pred_tensor[...,:4]
        pred_scores=pred_tensor[...,4:5]
        pred_classes=pred_tensor[...,5:6]
        pred_scores=tf.squeeze(pred_scores,axis=-1)
        nms_index=tf.image.non_max_suppression(pred_box,
                                               pred_scores,
                                               max_bboxes,
                                               iou_threshold=iou_thres,
                                               score_threshold=score_thres)
        pred_box=tf.gather(pred_box,nms_index)
        pred_scores=tf.gather(pred_scores,nms_index)
        pred_scores=tf.expand_dims(pred_scores,axis=-1)
        pred_classes=tf.gather(pred_classes,nms_index)
        pred_tensor=tf.concat([pred_box,pred_scores,pred_classes],axis=-1)
        return pred_tensor
    @tf.Module.with_name_scope
    def _SoftNMS(self,pred_tensor,max_bboxes=100,iou_thres=0.5,score_thres=0.01):
        pred_box=pred_tensor[...,:4]
        pred_scores=pred_tensor[...,4:5]
        pred_classes=pred_tensor[...,5:6]
        pred_scores=tf.squeeze(pred_scores,axis=-1)
        nms_index,pred_scores=tf.image.non_max_suppression_with_scores(pred_box,
                                                                       pred_scores,
                                                                       max_bboxes,
                                                                       iou_threshold=iou_thres,
                                                                       score_threshold=score_thres,
                                                                       soft_nms_sigma=0.5)
        pred_box=tf.gather(pred_box,nms_index)
        pred_scores=tf.expand_dims(pred_scores,axis=-1)
        pred_classes=tf.gather(pred_classes,nms_index)
        pred_tensor=tf.concat([pred_box,pred_scores,pred_classes],axis=-1)
        return pred_tensor
    @tf.Module.with_name_scope
    def _CategoryNMS(self,pred_tensor):
        pred_box=pred_tensor[...,:4]
        pred_scores=pred_tensor[...,4:5]
        pred_classes=pred_tensor[...,5:6]
        for i in range(self._labels_len):
            pred_mask=pred_classes==i
            pred_mask=tf.squeeze(pred_mask,-1)
            _pred_box=tf.boolean_mask(pred_box,pred_mask)
            _pred_scores=tf.boolean_mask(pred_scores,pred_mask)
            _pred_classes=tf.boolean_mask(pred_classes,pred_mask)
            _pred_tensor=tf.concat([_pred_box,_pred_scores,_pred_classes],axis=-1)
            _pred_tensor=self._SoftNMS(_pred_tensor,self._max_boxes_per_cls,self._iou_threshold,self._score_threshold)
            if(i==0):final_pred_tensor=_pred_tensor
            else:final_pred_tensor=tf.concat([final_pred_tensor,_pred_tensor],axis=0)
        return final_pred_tensor
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        output_ts_list=[]
        for i,input_ts in enumerate(input_ts_list):
            output_ts=self._PreprocessPredTs(input_ts)
            output_ts_list.append(output_ts)
        output_ts=tf.concat(output_ts_list,axis=0)
        if(self._nms_type=="category_nms"):
            output_ts=self._CategoryNMS(output_ts)
        elif(self._nms_type=="nms"):
            output_ts=self._NMS(output_ts,self._max_boxes_per_cls,self._iou_threshold,self._score_threshold)
        elif(self._nms_type=="soft_nms"):
            output_ts=self._SoftNMS(output_ts,self._max_boxes_per_cls,self._iou_threshold,self._score_threshold)
        output_ts=self._PostProcessPredTs(output_ts)
        return output_ts

class CSLYOLOBody(tf.Module):
    def __init__(self,fpn_filters=96,repeat=3,name="cslyolobody"):
        super(CSLYOLOBody,self).__init__(name=name)
        self._fpn_filters=round(fpn_filters)
        self._repeat=round(repeat)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._l1_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l1_cspg")
        self._l3_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l3_cspg")
        self._l5_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l5_cspg")
        self._l2_bifusion=InputBIFusion(name=self._name+"_l2_bifusion")
        self._l4_bifusion=InputBIFusion(name=self._name+"_l4_bifusion")

        self._cslfpn=CSLFPN(repeat=self._repeat,name=self._name+"_cslfpn")
        # self._vanillafpn=VanillaFPN(name=self._name+"_vanillafpn")
    @tf.Module.with_name_scope
    def __call__(self,bacbone_l1,bacbone_l2,bacbone_l3):
        orig_l1,orig_l2,orig_l3=bacbone_l1,bacbone_l2,bacbone_l3
        l1=self._l1_cspg(orig_l1)
        l3=self._l3_cspg(orig_l2)
        l5=self._l5_cspg(orig_l3)
        l2=self._l2_bifusion(l1,l3)
        l4=self._l4_bifusion(l3,l5)
        l1,l2,l3,l4,l5=self._cslfpn([l1,l2,l3,l4,l5])
        # l1,l2,l3,l4,l5=self._vanillafpn([l1,l2,l3,l4,l5])
        return l1,l2,l3,l4,l5

class CSLLoss(tf.Module):
    def __init__(self,name="cslloss"):
        super(CSLLoss,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _TrueWeight(self,true_y):
        true_wht=tf.squeeze(true_y[...,8:9],-1)
        return true_wht
    @tf.Module.with_name_scope
    def _TrueMask(self,true_y):
        true_mask=tf.cast(tf.squeeze(true_y[...,8:9],-1)>0.,tf.float32)
        return true_mask
    @tf.Module.with_name_scope
    def _IgnoreMask(self,true_y,pred_y,true_mask):
        pred_boxes=pred_y[...,4:8]
        true_boxes=true_y[...,4:8]

        pred_xy=pred_boxes[...,0:2]
        pred_wh=pred_boxes[...,2:4]
        pred_xy=tf.clip_by_value(pred_xy,0.0,1.0)
        pred_wh=tf.clip_by_value(pred_wh,1e-8,1.0)
        pred_xy=tf.expand_dims(pred_xy,4)
        pred_wh=tf.expand_dims(pred_wh,4)
        pred_wh_half=pred_wh/2.
        pred_mins=pred_xy-pred_wh_half
        pred_maxes=pred_xy+pred_wh_half
            
        mask_true_boxes=tf.boolean_mask(true_boxes,tf.cast(true_mask,tf.bool))
        mask_true_boxes_shape=tf.shape(mask_true_boxes)
        mask_true_boxes=tf.reshape(mask_true_boxes,[1,1,1,1,mask_true_boxes_shape[0],4])

        true_xy=mask_true_boxes[...,0:2]
        true_wh=mask_true_boxes[...,2:4]
        true_wh_half=true_wh/2.
        true_mins=true_xy-true_wh_half
        true_maxes=true_xy+true_wh_half
        intersect_mins=tf.maximum(pred_mins,true_mins)
        intersect_maxes=tf.minimum(pred_maxes,true_maxes)
        intersect_wh=tf.maximum(intersect_maxes-intersect_mins,tf.zeros_like(intersect_maxes))
        intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
        pred_areas=pred_wh[...,0]*pred_wh[...,1]
        true_areas=true_wh[...,0]*true_wh[...,1]
        union_area=pred_areas+true_areas-intersect_area
        iou=tf.math.divide_no_nan(intersect_area,union_area)
        best_iou=tf.reduce_max(iou,axis=4)
        ignore_mask=tf.cast(best_iou<0.5,tf.float32)
        return ignore_mask
    @tf.Module.with_name_scope
    def _CIoU(self,true_y,pred_y,true_mask,true_wht):
        pred_xy=pred_y[...,4:6]
        pred_wh=pred_y[...,6:8]
        pred_xy=tf.clip_by_value(pred_xy,0.0,1.0)
        pred_wh=tf.clip_by_value(pred_wh,1e-8,1.0)
        pred_wh_half=pred_wh/2.
        pred_mins=pred_xy-pred_wh_half
        pred_maxes=pred_xy+pred_wh_half
        
        true_xy=true_y[...,4:6]
        true_wh=true_y[...,6:8]
        true_wh_half=true_wh/2.
        true_mins=true_xy-true_wh_half
        true_maxes=true_xy+true_wh_half

        intersect_mins=tf.maximum(pred_mins,true_mins)
        intersect_maxes=tf.minimum(pred_maxes,true_maxes)
        intersect_wh=tf.maximum(intersect_maxes-intersect_mins,tf.zeros_like(intersect_maxes))
        intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
        pred_areas=pred_wh[...,0]*pred_wh[...,1]
        true_areas=true_wh[...,0]*true_wh[...,1]
        union_area=pred_areas+true_areas-intersect_area
        iou=tf.math.divide_no_nan(intersect_area,union_area)

        enclose_left_up=tf.minimum(pred_mins,true_mins)
        enclose_right_down=tf.maximum(pred_maxes,true_maxes)
        enclose_section=tf.maximum(enclose_right_down-enclose_left_up,tf.zeros_like(enclose_right_down))
        enclose_c2=enclose_section[...,0]**2+enclose_section[...,1]**2

        p2=(pred_xy[...,0]-true_xy[...,0])**2+(pred_xy[...,1]-true_xy[...,1])**2

        atan1=tf.atan(pred_wh[...,0]/pred_wh[...,1])
        temp_a=tf.keras.backend.switch(true_wh[...,1]>0.0,true_wh[...,1],true_wh[...,1]+1.0)
        atan2=tf.atan(true_wh[...,0]/temp_a)
        v=4.0*(atan1-atan2)**2/(math.pi**2)
        a=v/(1-iou+v)
        ciou=iou-1.0*p2/enclose_c2-1.0*a*v

        ciou=tf.expand_dims(ciou,axis=-1)
        coord_loss_scale=2-true_wh[...,0]*true_wh[...,1]
        ciou_loss=coord_loss_scale*true_mask*true_wht*tf.reduce_sum((1.0-ciou),axis=-1)
        ciou_loss=tf.reduce_sum(ciou_loss,axis=(1,2,3))
        return ciou_loss
    @tf.Module.with_name_scope
    def _BboxesLoss(self,true_y,pred_y,true_mask,true_wht):
        box_for_fit_true=true_y[...,:4]
        box_for_fit_pred=pred_y[...,:4]
        true_wh=true_y[...,6:8]
        coord_loss_scale=2-true_wh[...,0]*true_wh[...,1]

        xy_loss=tf.keras.backend.binary_crossentropy(box_for_fit_true[...,:2],box_for_fit_pred[...,:2])

        #smooth l1
        huber_delta=0.5
        wh_loss=tf.math.abs(box_for_fit_true[...,2:]-box_for_fit_pred[...,2:])
        wh_loss=tf.keras.backend.switch(wh_loss<huber_delta,0.5*wh_loss**2,huber_delta*(wh_loss-0.5*huber_delta))
    
        coord_loss=tf.reduce_sum(xy_loss+wh_loss,axis=-1)

        coord_loss=coord_loss_scale*true_mask*true_wht*coord_loss
        coord_loss=tf.reduce_sum(coord_loss,axis=(1,2,3))
        return coord_loss
    @tf.Module.with_name_scope
    def _ConfidenceLoss(self,pred_y,true_mask,ignore_mask,true_wht):
        pred_cnfd=pred_y[...,8:9]
        pstv_bce_loss=tf.keras.backend.binary_crossentropy(tf.ones_like(pred_cnfd),pred_cnfd)
        ngtv_bce_loss=tf.keras.backend.binary_crossentropy(tf.zeros_like(pred_cnfd),pred_cnfd)
        pstv_loss=true_mask*true_wht*tf.reduce_sum(pstv_bce_loss,axis=-1)
        ngtv_loss=(1-true_mask)*ignore_mask*tf.reduce_sum(ngtv_bce_loss,axis=-1)
        cnfd_loss=pstv_loss+ngtv_loss
        cnfd_loss=tf.reduce_sum(cnfd_loss,axis=(1,2,3))
        return cnfd_loss
    @tf.Module.with_name_scope
    def _ConfidenceFocalLoss(self,pred_y,true_mask,ignore_mask,true_wht,alpha=0.5,gamma=1.5):
        pred_cnfd=pred_y[...,8:9]
        pstv_fctr=alpha
        ngtv_fctr=1-alpha
        pstv_wht=pstv_fctr*(1-pred_cnfd)**gamma
        ngtv_wht=ngtv_fctr*pred_cnfd**gamma
        pstv_bce_loss=tf.keras.backend.binary_crossentropy(tf.ones_like(pred_cnfd),pred_cnfd)
        ngtv_bce_loss=tf.keras.backend.binary_crossentropy(tf.zeros_like(pred_cnfd),pred_cnfd)
        pstv_loss=true_mask*true_wht*tf.reduce_sum(pstv_wht*pstv_bce_loss,axis=-1)
        ngtv_loss=(1-true_mask)*ignore_mask*tf.reduce_sum(ngtv_wht*ngtv_bce_loss,axis=-1)
        cnfd_loss=pstv_loss+ngtv_loss
        cnfd_loss=tf.reduce_sum(cnfd_loss,axis=(1,2,3))
        return cnfd_loss
    @tf.Module.with_name_scope
    def _ClassesLoss(self,true_y,pred_y,true_mask,true_wht):
        true_cls=true_y[...,10:]
        pred_cls=pred_y[...,9:]
        bce_loss=tf.keras.backend.binary_crossentropy(true_cls,pred_cls)
        cls_loss=true_mask*true_wht*tf.reduce_sum(bce_loss,axis=-1)
        cls_loss=tf.reduce_sum(cls_loss,axis=(1,2,3))
        return cls_loss
    @tf.Module.with_name_scope
    def __call__(self):
        def _CSLLoss(true_y,pred_y):
            true_mask=self._TrueMask(true_y)
            ignore_mask=self._IgnoreMask(true_y,pred_y,true_mask)
            true_wht=self._TrueWeight(true_y)

            iou_loss=self._CIoU(true_y,pred_y,true_mask,true_wht)
            coord_loss=self._BboxesLoss(true_y,pred_y,true_mask,true_wht)
            cnfd_loss_1=self._ConfidenceLoss(pred_y,true_mask,ignore_mask,true_wht)
            cnfd_loss=self._ConfidenceFocalLoss(pred_y,true_mask,ignore_mask,true_wht)
            classes_loss=self._ClassesLoss(true_y,pred_y,true_mask,true_wht)

            loss=iou_loss+coord_loss+cnfd_loss_1+cnfd_loss+classes_loss
            return loss
        return _CSLLoss
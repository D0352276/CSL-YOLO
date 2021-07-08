import tensorflow as tf
from cslyolo import CSLConv,CSLHead,CSLYOLOBody,CSLLoss
from cslbone import CSLBone
from tools import FreezeLayers,UnfreezeLayers
_strategy=tf.distribute.MirroredStrategy()

def CSLBoneBody(input_ts,freeze=False):
    bacbone_outputs=CSLBone()(input_ts)
    model=tf.keras.Model(input_ts,bacbone_outputs[-1])
    model.load_weights("weights/cslb_whts.hdf5")
    if(freeze==True):FreezeLayers(model,freeze_type="ALL")
    return bacbone_outputs

def MobileNetV2(input_ts,alpha=1.0,freeze=False):
    global _strategy
    with _strategy.scope():
        model=tf.keras.applications.MobileNetV2(input_tensor=input_ts,alpha=alpha,include_top=False,weights=None)
        if(freeze==True):FreezeLayers(model,freeze_type="ALL")
        l1=model.layers[-101].output
        l2=model.layers[-39].output
        l3=model.layers[-12].output
    return l1,l2,l3

def EffictienNetB0(input_ts,freeze=False):
    import efficientnet.tfkeras as efn
    global _strategy
    with _strategy.scope():
        model=efn.EfficientNetB0(input_tensor=input_ts,include_top=False,weights='imagenet')
        model=tf.keras.Model(input_ts,model.layers[-4].output)
        if(freeze==True):FreezeLayers(model,freeze_type="ALL")
        l1=model.layers[-158].output
        l2=model.layers[-72].output
        l3=model.layers[-14].output
    return l1,l2,l3

def EffictienNetB1(input_ts,freeze=False):
    import efficientnet.tfkeras as efn 
    global _strategy
    with _strategy.scope():
        model=efn.EfficientNetB1(input_tensor=input_ts,include_top=False,weights='imagenet')
        model=tf.keras.Model(input_ts,model.layers[-4].output)
        if(freeze==True):FreezeLayers(model,freeze_type="ALL")
        l1=model.layers[-218].output
        l2=model.layers[-102].output
        l3=model.layers[-29].output
    return l1,l2,l3

def CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters=64,fpn_repeat=3,backbone="cslb",freeze=False):
    global _strategy
    with _strategy.scope():
        input_ts=tf.keras.Input(shape=input_shape)
        if(backbone=="b0"):
            bacbone_outputs=EffictienNetB0(input_ts,freeze=freeze)
        elif(backbone=="b1"):
            bacbone_outputs=EffictienNetB1(input_ts,freeze=freeze)
        elif(backbone=="m2"):
            bacbone_outputs=MobileNetV2(input_ts,freeze=freeze)
        elif(backbone=="cslb"):
            bacbone_outputs=CSLBoneBody(input_ts,freeze=freeze)
        body_outputs=CSLYOLOBody(fpn_filters,fpn_repeat)(*bacbone_outputs)
        net_outputs=CSLConv(anchors_list[0:],labels_len,name="cslconv")(body_outputs[0:])

        model=tf.keras.Model(input_ts,net_outputs)
    return model

def CompileCSLYOLO(model,heads_len,whts_path=None,lr=0.0001,compile_type="train"):
    global _strategy
    with _strategy.scope():
        if(whts_path!=None):
            model.load_weights(whts_path)
        if(compile_type=="train"):
            losses=[CSLLoss(name="cslloss_"+str(i))() for i in range(heads_len)]
            loss_weights=[1/heads_len for i in range(heads_len)]
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss=losses,
                        loss_weights=loss_weights)
    return model

def CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=10,score_thres=0.5,iou_thres=0.5,nms_type="category_nms"):
    input_ts=model.layers[0].output
    orig_hw=model.layers[0].get_input_shape_at(0)[1:3]

    heads_ts=[]
    for i in range(heads_len,0,-1):
        heads_ts.append(model.layers[-i].output)
    output_op=CSLHead(orig_hw,labels_len,max_boxes_per_cls,score_thres,iou_thres,nms_type=nms_type)(heads_ts)
    model=tf.keras.Model(input_ts,output_op)
    return model

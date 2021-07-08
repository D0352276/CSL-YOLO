import numpy as np
import cv2
import os
from tools import Bboxes2JSON,ResizeImg,DeTransformBboxes,InitLabels2bgrDict,Drawing

def Training(model,train_data,validation_data=None,batch_size=1,epochs=1,step_per_epoch=1,callbacks=[]):
    def gen():yield 1
    if(type(train_data)==type(gen())):
        model.fit(train_data,
                  validation_data=validation_data,
                  epochs=epochs,
                  steps_per_epoch=step_per_epoch,
                  max_queue_size=32,
                  workers=1,
                  shuffle=False,
                  use_multiprocessing=False,
                  callbacks=callbacks)
    elif(type(train_data)==list or type(train_data)==tuple):
        model.fit(train_data,
                  validation_data=validation_data,
                  epochs=epochs,
                  callbacks=callbacks)

def Predicting(model,labels,img):
    orig_img_hw=np.shape(img)[:2]
    output_hw=np.array(model.layers[0].get_input_shape_at(0)[1:3])
    wh_ratio=np.flip(orig_img_hw/output_hw,axis=-1)
    img=cv2.resize(img,(output_hw[1],output_hw[0]))
    img=img/255
    img=np.array([img])
    pred_msg=model.predict_on_batch(img)
    if(np.shape(pred_msg)[0]==0):return np.array([])
    pred_boxes=(pred_msg[...,:4]*np.concatenate([wh_ratio,wh_ratio],axis=-1)).astype("float")
    pred_boxes=np.around(pred_boxes,decimals=1)
    pred_scores=pred_msg[...,4:5]
    pred_classes=pred_msg[...,5:]
    pred_classes=pred_classes.tolist()
    pred_classes=list(map(lambda x:[labels[int(x[0])]],pred_classes))
    pred_classes=np.array(pred_classes)
    pred_bboxes=np.concatenate([pred_boxes,pred_scores,pred_classes],axis=-1)
    return pred_bboxes

def PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=False,printing=False,img_type="jpg"):
    if(drawing==True):InitLabels2bgrDict(labels)
    imgs_name=os.listdir(imgs_dir)
    for i,img_name in enumerate(imgs_name):
        try:
            name,_type=img_name.split(".")
            if(_type!=img_type):continue
        except:continue
        img=cv2.imread(imgs_dir+"/"+img_name)
        pred_bboxes=Predicting(model,labels,img)
        if(drawing==True):
            img=Drawing(img,pred_bboxes)
            cv2.imwrite(pred_dir+"/"+img_name,img)
        Bboxes2JSON(pred_bboxes,pred_dir+"/json/"+name+".json")
        if(printing==True):print(str(i)+" Predicting Done.")
    return 
import argparse
from tools import ParsingCfg
import numpy as np
from tools import InitDataDir

# ###
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ###


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("cfg_path",help="config file path",type=str)
    parser.add_argument("-t","--train",help="training mode",action="store_true")
    parser.add_argument("-ce","--cocoevaluation",help="coco evaluation mode",action="store_true")
    parser.add_argument("-e","--evaluation",help="evaluation mode",action="store_true")
    parser.add_argument("-fe","--fpsevaluation",help="fps evaluation mode",action="store_true")
    parser.add_argument("-p","--predict",help="prediction mode",action="store_true")
    parser.add_argument("-d","--demo",help="demo mode",action="store_true")
    parser.add_argument("-td","--tflitedemo",help="tflite demo mode",action="store_true")
    parser.add_argument("-trcd","--trackingdemo",help="tracking demo mode",action="store_true")
    parser.add_argument("-cvt","--convert",help="tf to tflite",action="store_true")
    parser.add_argument("-s","--server",help="sever mode",action="store_true")
    args=parser.parse_args()

    mode="train"
    cfg_path=args.cfg_path
    if(args.train==True):mode="train"
    elif(args.cocoevaluation==True):mode="cocoevaluation"
    elif(args.evaluation==True):mode="evaluation"
    elif(args.fpsevaluation==True):mode="fpsevaluation"
    elif(args.predict==True):mode="predict"
    elif(args.demo==True):mode="demo"
    elif(args.tflitedemo==True):mode="tflitedemo"
    elif(args.trackingdemo==True):mode="trackingdemo"
    elif(args.convert==True):mode="convert"
    elif(args.server==True):mode="server"

    cfg_dict=ParsingCfg(cfg_path)

    input_shape=list(map(lambda x:int(x),cfg_dict["input_shape"]))
    out_hw_list=list(map(lambda x:[int(x[0]),int(x[1])],cfg_dict["out_hw_list"]))
    heads_len=len(out_hw_list)

    backbone=cfg_dict["backbone"]
    fpn_filters=cfg_dict["fpn_filters"]
    fpn_repeat=cfg_dict["fpn_repeat"]

    l1_anchors=np.array(cfg_dict["l1_anchors"])
    l2_anchors=np.array(cfg_dict["l2_anchors"])
    l3_anchors=np.array(cfg_dict["l3_anchors"])
    l4_anchors=np.array(cfg_dict["l4_anchors"])
    l5_anchors=np.array(cfg_dict["l5_anchors"])

    anchors_list=[l1_anchors*np.array(out_hw_list[0]),
                  l2_anchors*np.array(out_hw_list[1]),
                  l3_anchors*np.array(out_hw_list[2]),
                  l4_anchors*np.array(out_hw_list[3]),
                  l5_anchors*np.array(out_hw_list[4])]
    anchoors_len=len(l1_anchors)
    labels=cfg_dict["labels"]
    labels_len=len(labels)


    if(mode=="train"):
        from create_model import CSLYOLO,CompileCSLYOLO
        from data_generator import DataGenerator,MultiDataGenerator
        from model_operation import Training
        from evaluate import CallbackEvalFunction
        from callbacks import Stabilizer,WeightsSaver,BestWeightsSaver

        init_weight=cfg_dict.get("init_weight_path",None)
        weight_save_path=cfg_dict["weight_save_path"]
        best_weight_save_path=cfg_dict["best_weight_save_path"]
        freeze=cfg_dict.get("freeze",False)

        #Must Contains Jsons
        train_dir=cfg_dict["train_dir"]
        valid_dir=cfg_dict["valid_dir"]
        pred_dir=cfg_dict["pred_dir"]

        batch_size=int(cfg_dict["batch_size"])
        step_per_epoch=int(cfg_dict["step_per_epoch"])
        epochs_schedule=list(map(lambda x:int(x),cfg_dict["epochs_schedule"]))
        lr_schedule=cfg_dict["lr_schedule"]
        callbacks_schedule=cfg_dict["callbacks_schedule"]

        gen=MultiDataGenerator(train_dir,train_dir+"/json",input_shape[:2],out_hw_list,anchors_list,labels,batch_size=batch_size,print_bool=False)
        gen.Start()

        stabilizer=Stabilizer()
        weight_saver=WeightsSaver(weight_save_path)
        best_weight_saver=BestWeightsSaver(best_weight_save_path,CallbackEvalFunction,eval_parms=[labels,valid_dir,pred_dir])

        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone,freeze)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=init_weight,lr=0.1)
        for i,epochs in enumerate(epochs_schedule):
            callbacks=[]
            lr=lr_schedule[i]
            for callback_name in callbacks_schedule[i]:
                if(callback_name=="stabilizer"):callbacks.append(stabilizer)
                if(callback_name=="weight_saver"):callbacks.append(weight_saver)
                if(callback_name=="best_weight_saver"):callbacks.append(best_weight_saver)
            model=CompileCSLYOLO(model,heads_len,whts_path=None,lr=lr,compile_type="train")
            Training(model,gen.Generator(),batch_size=batch_size,epochs=epochs,step_per_epoch=step_per_epoch,callbacks=callbacks)
        gen.Stop()
    elif(mode=="predict"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingImgs
        
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        InitDataDir(pred_dir)
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        drawing=cfg_dict["drawing"]
        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type)
        PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=drawing,printing=True)
    elif(mode=="cocoevaluation"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingImgs
        from evaluate.cocoeval import COCOEval
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        annotation_path=cfg_dict["annotation_path"]
        label2id_path=cfg_dict["label2id_path"]
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        overwrite=cfg_dict["overwrite"]
        if(overwrite==True):
            InitDataDir(pred_dir)
            model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
            model.summary()
            model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
            model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type)
            PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=False,printing=True)
        COCOEval(annotation_path,pred_dir+"/json",label2id_path)
    elif(mode=="evaluation"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingImgs
        from evaluate import Evaluation
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        test_dir=cfg_dict["test_dir"]
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        overwrite=cfg_dict["overwrite"]
        if(overwrite==True):
            InitDataDir(pred_dir)
            model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
            model.summary()
            model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
            model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type)
            PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=False,printing=True)
        mean_ap=Evaluation(labels,test_dir,pred_dir)
        print("mAP: "+str(mean_ap))
    elif(mode=="fpsevaluation"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from evaluate import FramePerSecond
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type)
        fps=FramePerSecond(model,input_shape)
        print("FPS: "+str(fps))
    elif(mode=="demo"):
        import cv2
        from tools import InitLabels2bgrDict,Drawing
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import Predicting
        from camera_stream import CameraStream
        import datetime
        import os
        
        weight_path=cfg_dict["weight_path"]
        videos_idx=int(cfg_dict["videos_idx"])
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]

        InitLabels2bgrDict(labels)

        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type="nms")

        camera_stream=CameraStream(videos_idx,show=True,save_dir="dataset/tracking")
        camera_stream.Start()
        while(camera_stream.StopChecking()==False):
            frame=camera_stream.GetFrame()
            pred_bboxes=Predicting(model,labels,frame)
            camera_stream.UpdateBboxes(pred_bboxes)
        camera_stream.Stop()
    elif(mode=="tflitedemo"):
        import cv2
        from tools import InitLabels2bgrDict,Drawing
        from model_operation import Predicting
        from tflite.tflite_cslyolo import TFLiteCSLYOLOBody,TFLiteCSLYOLOHead,TFLiteCSLYOLOPredicting
        import datetime
        
        tflite_path=cfg_dict["tflite_path"]
        videos_idx=int(cfg_dict["videos_idx"])
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        
        
        InitLabels2bgrDict(labels)
        interpreter=TFLiteCSLYOLOBody(tflite_path)
        tfl_head_model=TFLiteCSLYOLOHead(input_shape[:2],out_hw_list,anchoors_len,labels_len,
                                         max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres)

        cap=cv2.VideoCapture(videos_idx)
        if not cap.isOpened():
            print("Cannot open camera.")
            exit()

        start_time=None
        frame_count=0
        while(True):
            if(start_time==None):start_time=datetime.datetime.now()
            frame_count+=1
            ret,frame=cap.read()
            pred_bboxes=TFLiteCSLYOLOPredicting(interpreter,tfl_head_model,labels,frame)
            frame=Drawing(frame,pred_bboxes)
            cv2.imshow('TFLDEMO',frame)
            if cv2.waitKey(1) == ord('q'):
                end_time=datetime.datetime.now()
                print("FPS: "+str(frame_count/(end_time-start_time).seconds))
                break
        cap.release()
        cv2.destroyAllWindows()
    elif(mode=="trackingdemo"):
        import cv2
        from tools import InitLabels2bgrDict,Drawing
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import Predicting
        
        p2_l1_anchors=np.array(cfg_dict["p2_l1_anchors"])
        p2_l2_anchors=np.array(cfg_dict["p2_l2_anchors"])
        p2_l3_anchors=np.array(cfg_dict["p2_l3_anchors"])
        p2_l4_anchors=np.array(cfg_dict["p2_l4_anchors"])
        p2_l5_anchors=np.array(cfg_dict["p2_l5_anchors"])

        p2_anchors_list=[p2_l1_anchors*np.array(out_hw_list[0]),
                         p2_l2_anchors*np.array(out_hw_list[1]),
                         p2_l3_anchors*np.array(out_hw_list[2]),
                         p2_l4_anchors*np.array(out_hw_list[3]),
                         p2_l5_anchors*np.array(out_hw_list[4])]
        p2_labels=cfg_dict["p2_labels"]

        weight_path=cfg_dict["weight_path"]
        p2_weight_path=cfg_dict["p2_weight_path"]
        videos_idx=int(cfg_dict["videos_idx"])
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]

        InitLabels2bgrDict(labels+p2_labels)

        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type="nms")

        p2_model=CSLYOLO(input_shape,p2_anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        p2_model.summary()
        p2_model=CompileCSLYOLO(p2_model,heads_len,whts_path=p2_weight_path,compile_type="predict")
        p2_model=CSLYOLOHead(p2_model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type="nms")

        cap=cv2.VideoCapture(videos_idx)
        if not cap.isOpened():
            print("Cannot open camera.")
            exit()
        while(True):
            ret,frame=cap.read()
            pred_bboxes=Predicting(model,labels,frame)
            p2_pred_bboxes=Predicting(p2_model,p2_labels,frame)
            frame=Drawing(frame,pred_bboxes)
            frame=Drawing(frame,p2_pred_bboxes)
            cv2.imshow('DEMO',frame)
            if(cv2.waitKey(1)==ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif(mode=="convert"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from tflite.converter import TFLiteConverter

        weight_path=cfg_dict["weight_path"]
        tflite_path=cfg_dict["tflite_path"]

        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        TFLiteConverter(model,tflite_path)
    # elif(mode=="server"):
    #     from create_model import CSLNet,CompileCSLNet,CSLNetHead
    #     from model_operation import PredictingImgs
    #     from server import CSLYServer
    #     import time

    #     weight_path=cfg_dict["weight_path"]
    #     max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
    #     score_thres=cfg_dict["score_thres"]
    #     iou_thres=cfg_dict["iou_thres"]
    #     model=CSLNet(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,backbone)
    #     model.summary()
    #     model=CompileCSLNet(model,heads_len,whts_path=weight_path)
    #     model=CSLNetHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres)

    #     csly_server=CSLYServer(model,labels,host="140.115.51.108")
    #     csly_server.Start()
    #     print("Initializing Server....Done.")
    #     time.sleep(999999)
    #     csly_server.Stop()
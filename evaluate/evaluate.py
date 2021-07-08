from .map import EvalPredBbox,mAP
from create_model import CSLYOLOHead

def Evaluation(labels,test_dir,pred_dir,iou_thres=0.5):
    result_dict=EvalPredBbox(test_dir,pred_dir,labels,iou_thres)
    mean_ap=mAP(result_dict,labels)
    return mean_ap

def CallbackEvalFunction(model,labels,test_dir,pred_dir,max_boxes_per_cls=5,score_thres=0.1):
    try:
        model=CSLYOLOHead(model,len(labels),max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres)
        mean_ap=Evaluation(model,labels,test_dir,pred_dir)
    except:mean_ap=0
    print("\n\nmAP: "+str(mean_ap)+"\n")
    return mean_ap
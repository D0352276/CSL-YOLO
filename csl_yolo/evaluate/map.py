import os
import numpy as np
from tools import JSON2Bboxes,IOU

def GetBboxes(pred_dir,test_dir,pred_file):
    pred_json_path=pred_dir+"/json/"+pred_file+".json"
    test_json_path=test_dir+"/json/"+pred_file+".json"
    pred_bboxes=JSON2Bboxes(pred_json_path)
    pred_bboxes.sort(key=lambda x:float(x[4]),reverse=True)
    true_bboxes=JSON2Bboxes(test_json_path)
    return pred_bboxes,true_bboxes

def EvalPredBbox(test_dir,pred_dir,labels,iou_threshold=0.5):
    pred_result_dict={}
    for label in labels:
        pred_result_dict[label]={}
        pred_result_dict[label]["result"]=[]
        pred_result_dict[label]["tot_true_num"]=0
    pred_files=os.listdir(pred_dir+"/json")
    pred_files=list(map(lambda x:x.split(".")[0],pred_files))
    for pred_file in pred_files:
        pred_bboxes,true_bboxes=GetBboxes(pred_dir,test_dir,pred_file)
        pred_bboxes=np.array(pred_bboxes)
        true_bboxes=np.array(true_bboxes)
        true_labels=true_bboxes[...,5]
        true_bboxes=true_bboxes[...,:4].astype("float").astype("int")
        for true_label in true_labels:
            pred_result_dict[true_label]["tot_true_num"]+=1
        if(len(pred_bboxes)==0):continue
        pred_scores=pred_bboxes[...,4].astype("float")
        pred_labels=pred_bboxes[...,5]
        pred_bboxes=pred_bboxes[...,:4].astype("float").astype("int")
        for j,pred_bbox in enumerate(pred_bboxes):
            iou_list=list(map(lambda b:IOU(b,pred_bbox),true_bboxes))
            try:
                best_iou=max(iou_list)
                best_idx=iou_list.index(best_iou)
                true_label=true_labels[best_idx]
            except:
                best_iou=0
                true_label=None
            pred_label=pred_labels[j]
            if(best_iou>=iou_threshold \
                and pred_label==true_label):
                result_bool=True
                true_labels[best_idx]=None
            else:
                result_bool=False
            pred_result_dict[pred_label]["result"].append([pred_scores[j],result_bool])
    for label in labels:
        pred_result_dict[label]["result"].sort(key=lambda x:x[0],reverse=True)
    return pred_result_dict

def RankPrecision(result_dict,label,rank):
    result_list=result_dict[label]["result"]
    tot_true_num=result_dict[label]["tot_true_num"]
    if(tot_true_num==0):return None
    result_list=result_list[:rank]
    tot_correct_num=0
    tot_pred_box_num=0
    for result in result_list:
        score,result_bool=result
        if(result_bool==True):tot_correct_num+=1
        tot_pred_box_num+=1
    return tot_correct_num/tot_pred_box_num

def RankRecall(result_dict,label,rank):
    result_list=result_dict[label]["result"]
    tot_true_num=result_dict[label]["tot_true_num"]
    if(tot_true_num==0):return None
    result_list=result_list[:rank]
    tot_correct_num=0
    for result in result_list:
        score,result_bool=result
        if(result_bool==True):tot_correct_num+=1
    return tot_correct_num/tot_true_num

def AveragePrecision(result_dict,label):
    top_rank=len(result_dict[label]["result"])
    if(top_rank==0):return 0
    ranks_recall=[]
    ranks_precision=[]
    for rank in range(1,top_rank+1):
        precision=RankPrecision(result_dict,label,rank)
        recall=RankRecall(result_dict,label,rank)
        ranks_precision.append(precision)
        ranks_recall.append(recall)
    ap=0
    last_precision=ranks_precision[top_rank-1]
    last_recall=ranks_recall[top_rank-1]
    for rank in range(top_rank-2,0,-1):
        cur_precision=ranks_precision[rank]
        cur_recall=ranks_recall[rank]
        delta_recall=last_recall-cur_recall
        last_recall=cur_recall
        if(cur_precision>last_precision):
            last_precision=cur_precision
        ap+=last_precision*delta_recall
    return ap

def mAP(result_dict,labels):
    ap_sum=0
    for label in labels:
        if(label=="face"):continue
        ap_sum+=AveragePrecision(result_dict,label)
    mean_ap=ap_sum/len(labels)
    return mean_ap
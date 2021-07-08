import json
import os
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 

def JSON2Dict(json_path):
    with open(json_path,"r") as json_fin:
        json_dict=json.load(json_fin)
    return json_dict

def Dict2JSON(json_dict,json_path):
    with open(json_path,"w") as fout:
        json.dump(json_dict,fout,ensure_ascii=False) 
    return 

def Json2COCODicts(label2id_dict,json_dir,json_file_name):
    bboxes=JSON2Dict(json_dir+"/"+json_file_name)["bboxes"]
    img_id=int(json_file_name.split(".")[0])
    obj_dicts=[]
    for bbox in bboxes:
        obj_dict={}
        x,y,w,h,score,label=bbox
        label_id=label2id_dict[label]
        obj_dict["image_id"]=img_id
        obj_dict["category_id"]=label_id
        obj_dict["bbox"]=[float(x),float(y),float(w),float(h)]
        obj_dict["score"]=float(score)
        obj_dicts.append(obj_dict)
    return obj_dicts

def Jsons2COCODicts(label2id_dict,json_dir):
    obj_dicts=[]
    files_name=os.listdir(json_dir)
    for file_name in files_name:
        if(file_name.split(".")[1]!="json"):
            continue
        obj_dicts+=Json2COCODicts(label2id_dict,json_dir,file_name)
    return obj_dicts

def Jsons2COCOJson(label2id_path,json_dir,save_path):
    label2id_dict=JSON2Dict(label2id_path)
    obj_dicts=Jsons2COCODicts(label2id_dict,json_dir)
    Dict2JSON(obj_dicts,save_path)
    return

def COCOEval(gt_json_path,pred_jsons_dir,label2id_json_path):
    Jsons2COCOJson(label2id_json_path,pred_jsons_dir,"./coco_eval_temp.json")
    coco_gt=COCO(gt_json_path)
    coco_dt=coco_gt.loadRes("./coco_eval_temp.json")
    cocoEval=COCOeval(coco_gt,coco_dt,"bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
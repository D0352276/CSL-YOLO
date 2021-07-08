import json

def JSON2Bboxes(json_path):
    with open(json_path,"r") as json_fin:
        json_dict=json.load(json_fin)
    return json_dict["bboxes"]

def Bboxes2JSON(bboxes,json_path):
    try:bboxes=bboxes.tolist()
    except:pass
    json_dict={}
    json_dict["bboxes"]=bboxes
    with open(json_path,"w") as fout:
        json.dump(json_dict,fout,ensure_ascii=False) 
    return 
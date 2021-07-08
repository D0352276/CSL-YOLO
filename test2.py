import json
import os

def JSON2Dict(json_path):
    with open(json_path,"r") as json_fin:
        json_dict=json.load(json_fin)
    return json_dict


t_img_dict=JSON2Dict("image_info_test-dev2017.json")

imgs_name=[]
for img_name in t_img_dict["images"]:
    img_name=img_name["file_name"]
    imgs_name.append(img_name)



all_img_name=os.listdir("dataset/coco/test")

for img_name in all_img_name:
    if(img_name not in imgs_name):
        os.remove("dataset/coco/test/"+img_name)
    

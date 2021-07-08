import cv2
from tools import Bboxes2JSON


img_in_dir="/home/difvoice/widerface_dataset/WIDER_train/images"
img_save_dir="dataset/wider_face/imgs"
json_save_dir="dataset/wider_face/json"
annotation_path="/home/difvoice/widerface_dataset/wider_face_split/wider_face_train_bbx_gt.txt"


fin=open(annotation_path,"r")
text=fin.read()
fin.close()

text=text.split("\n")
del text[-1]
img=None
img_name=None
json_name=None
bboxes=[]
img_count=0

for i,text_line in enumerate(text):
    text_line=text_line.split(" ")
    if(len(text_line)==1):
        if(len(text[i+1].split(" "))==1):
            if(type(img)!=type(None)):
                img_count+=1
                print(img_count)
                cv2.imwrite(img_save_dir+"/"+img_name,img)
                Bboxes2JSON(bboxes,json_save_dir+"/"+json_name)
            img_name=text_line[0].split("/")[1]
            json_name=img_name.split(".")[0]+".json"
            img_path=img_in_dir+"/"+text_line[0]
            img=cv2.imread(img_path)
            bboxes=[]
        else:continue
    else:
        text_line=text_line[:-1]
        codes=list(map(lambda x:float(x),text_line))
        x,y,w,h,blur,expression,illumination,invalid,occlusion,pose=codes
        # if(occlusion!=0):continue
        bboxes.append([x,y,w,h,"face"])
cv2.imwrite(img_save_dir+"/"+img_name,img)
Bboxes2JSON(bboxes,json_save_dir+"/"+json_name)

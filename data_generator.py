import os
import cv2
import json
import random
import numpy as np
from aug import MixupAugment
from tools import JSON2Bboxes,IOU,ThreadPool
import queue
import time

class DataGenerator:
    def __init__(self,imgs_dir,jsons_dir,
                 img_hw,out_hw_list,anchors_list,
                 labels,batch_size=-1,data_type="jpg",
                 print_bool=True):
        self._imgs_dir=imgs_dir
        self._jsons_dir=jsons_dir
        self._files_name=np.array(list(map(lambda x:x.split(".")[0],os.listdir(self._jsons_dir))))
        self._files_len=len(self._files_name)
        self._img_hw=img_hw
        self._out_hw_list=out_hw_list
        self._anchors_list=anchors_list
        self._anchors_len=len(self._anchors_list[0])
        self._heads_len=len(self._anchors_list)
        self._labels=labels
        self._labels_len=len(labels)
        self._batch_size=batch_size
        self._data_type=data_type
        self._print_bool=print_bool
    def __call__(self):
        return self.Read()
    def _EncodingX(self,x):
        return x/255
    def _EncodingY(self,y):
        bboxes=y
        bboxes=self._PreprocessBBoxes(bboxes)
        ftmp_list=self._GetOutFtmpList()
        if(bboxes.shape[0]>0):
            bboxes_list=self._ScaleBBoxesByHWList(bboxes)
            ftmp_list=self._Bboxes2Ftmps(ftmp_list,bboxes_list)
        return ftmp_list
    def _GetImgAndBboxes(self,data_file):
        img_path=self._imgs_dir+"/"+data_file+"."+self._data_type
        json_path=self._jsons_dir+"/"+data_file+".json"
        img=cv2.imread(img_path)
        bboxes=JSON2Bboxes(json_path)
        return img,bboxes
    def _PreprocessBBoxes(self,true_bboxes):
        normalized_bboxes=[]
        for i,true_bbox in enumerate(true_bboxes):
            x,y,w,h,wht,label=true_bbox
            if(w<3 or h<3):continue
            x=x/self._img_hw[1]
            y=y/self._img_hw[0]
            w=w/self._img_hw[1]
            h=h/self._img_hw[0]
            x=np.clip(x,0,1.0)
            y=np.clip(y,0,1.0)
            w=np.clip(w,1e-8,1.0-x)
            h=np.clip(h,1e-8,1.0-y)

            label_onehot=np.zeros([self._labels_len+1])
            label_onehot[self._labels.index(label)]=1

            new_bbox=np.array([(x+w/2),(y+h/2),w,h,wht])
            new_bbox=np.concatenate([new_bbox,label_onehot],axis=-1)
            normalized_bboxes.append(new_bbox)
        return np.array(normalized_bboxes)
    def _GetOutFtmp(self,output_hw,anchors_len):
        out_ftmp=np.zeros([output_hw[0],
                           output_hw[1],
                           anchors_len,
                           4+6+self._labels_len+1])
        out_ftmp[...,-1]=1
        return out_ftmp
    def _ScaleBBoxes(self,true_bboxes,target_hw):
        true_bboxes=np.array(true_bboxes)
        true_xywh=true_bboxes[...,0:4]*np.reshape(np.array([target_hw[1],
                                                            target_hw[0],
                                                            target_hw[1],
                                                            target_hw[0]]),
                                                            [-1,4])
        true_bboxes=np.concatenate([true_xywh,true_bboxes[...,4:]],axis=-1)
        return true_bboxes
    def _Bbox2Ftmp(self,out_ftmp,true_bbox,anchors,use_best=False,iou_thres=0.2):
        output_hw=np.shape(out_ftmp)[:2]
        x,y,w,h=true_bbox[:4]
        norm_x=x/output_hw[1]
        norm_y=y/output_hw[0]
        norm_w=w/output_hw[1]
        norm_h=h/output_hw[0]
        wht=true_bbox[4]
        labels=true_bbox[5:]


        int_x=np.floor(x).astype('int')
        int_y=np.floor(y).astype('int')

        iou_list=list(map(lambda anchor:IOU([x,y,w,h],[int_x,int_y,anchor[0],anchor[1]]),anchors))
        for anchor_idx,iou in enumerate(iou_list):
            if(iou>iou_thres):
                if(int_x>=0 and int_x<output_hw[1] and int_y>=0 and int_y<output_hw[0] and \
                   iou>out_ftmp[int_y,int_x,anchor_idx,9]):
                    out_ftmp[int_y,int_x,anchor_idx,0]=x-int_x
                    out_ftmp[int_y,int_x,anchor_idx,1]=y-int_y
                    out_ftmp[int_y,int_x,anchor_idx,2]=w-anchors[anchor_idx][0]
                    out_ftmp[int_y,int_x,anchor_idx,3]=h-anchors[anchor_idx][1]
                    out_ftmp[int_y,int_x,anchor_idx,4]=norm_x
                    out_ftmp[int_y,int_x,anchor_idx,5]=norm_y
                    out_ftmp[int_y,int_x,anchor_idx,6]=norm_w
                    out_ftmp[int_y,int_x,anchor_idx,7]=norm_h
                    out_ftmp[int_y,int_x,anchor_idx,8]=wht
                    out_ftmp[int_y,int_x,anchor_idx,9]=iou
                    out_ftmp[int_y,int_x,anchor_idx,10:]=labels
        return
    def _ScaleBBoxesByHWList(self,bboxes):
        return [self._ScaleBBoxes(bboxes,hw) for hw in self._out_hw_list]
    def _GetOutFtmpList(self):
        return [self._GetOutFtmp(hw,self._anchors_len) for hw in self._out_hw_list]
    def _Bboxes2Ftmps(self,ftmp_list,bboxes_list):
        bboxes_buf=[0 for i in range(self._heads_len)]
        for i in range(self._heads_len):
            bboxes_buf[i]=bboxes_list[i]
        for i in range(len(bboxes_buf[0])):
            for j in range(self._heads_len):
                self._Bbox2Ftmp(ftmp_list[j],bboxes_buf[j][i],self._anchors_list[j],iou_thres=0.2)
        return ftmp_list
    def _GetRandomeFilesName(self,batch_size):
        batch_files=[]
        for i in range(batch_size):
            batch_files.append(self._files_name[random.randint(0,self._files_len-1)])
        return batch_files
    def Read(self):
        x=[]
        y_list=[[] for i in range(self._heads_len)]
        if(self._batch_size!=-1):
            files_name_1=self._GetRandomeFilesName(self._batch_size)
            files_name_2=self._GetRandomeFilesName(self._batch_size)
        for i,file_name_1 in enumerate(files_name_1):
            if(self._print_bool==True):print(str(i+1)+"th Loading "+file_name_1+"............",end="")
            file_name_2=files_name_2[i]
            img_1,bboxes_1=self._GetImgAndBboxes(file_name_1)
            img_2,bboxes_2=self._GetImgAndBboxes(file_name_2)
            img,bboxes=MixupAugment(img_1,bboxes_1,img_2,bboxes_2,self._img_hw)
            x.append(self._EncodingX(img))
            ftmp_list=self._EncodingY(bboxes)
            for j in range(self._heads_len):
                y_list[j].append(ftmp_list[j])
            if(self._print_bool==True):print("done.")
        for j in range(self._heads_len):
            y_list[j]=np.array(y_list[j])
        return np.array(x),tuple(y_list)
    def Generator(self):
        while 1:
            yield self.Read()

class MultiDataGenerator:
    def __init__(self,imgs_dir,jsons_dir,
                 img_hw,out_hw_list,anchors_list,
                 labels,batch_size=-1,thread_num=32,
                 max_queue=64,data_type="jpg",print_bool=True):
        self._data_gen=DataGenerator(imgs_dir,jsons_dir,img_hw,
                                     out_hw_list,anchors_list,labels,
                                     batch_size,data_type,print_bool)
        self._thread_pool=ThreadPool(thread_num)
        self._thread_num=thread_num
        self._batch_data_queue=queue.Queue()
        self._max_queue=max_queue
        self._stop_signal=False
    def __call__(self):
        return self.Read()
    def _ReadFunction(self):
        sleep_signal=True
        while(1):
            if(self._batch_data_queue.qsize()<self._max_queue//2):
                sleep_signal=False
            elif(self._batch_data_queue.qsize()>=self._max_queue and sleep_signal==False):
                sleep_signal=True
            if(sleep_signal==False):
                batch_data=self._data_gen()
                self._batch_data_queue.put(batch_data)
            else:
                time.sleep(0.01)
            if(self._stop_signal==True):break
    def Start(self):
        for i in range(self._thread_num):
            self._thread_pool.Push(self._ReadFunction)
        self._thread_pool.Start()
        return 
    def Stop(self):
        self._stop_signal=True
        self._thread_pool.Stop()
        return
    def Read(self):
        while(1):
            try:
                batch_data=self._batch_data_queue.get_nowait()
                break
            except:
                pass
        return batch_data
    def Generator(self):
        while(1):
            yield self.Read()

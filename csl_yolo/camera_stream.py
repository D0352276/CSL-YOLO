import cv2
import os
import time
from multiprocessing import Process,Queue
from tools import Drawing,DrawStatus,Bboxes2JSON

class CameraStream:
    def __init__(self,camera_idx,video_hw=[512,512],show=False,save_dir=""):
        self._camera_idx=camera_idx
        self._video_hw=video_hw
        self._show_bool=show
        self._save_dir=save_dir
        self._self2process_queue=Queue()
        self._process2self_queue=Queue()
        self._stopsignal_queue=Queue()
        self._process=self._InitProcess()
    def _InitProcess(self):
        init_dict={}
        init_dict["CAMERA_IDX"]=self._camera_idx
        init_dict["VIDEO_HW"]=self._video_hw
        init_dict["SHOW_BOOL"]=self._show_bool
        init_dict["SAVE_DIR"]=self._save_dir
        self._self2process_queue.put(init_dict)
        process=Process(target=self._FrameStream,args=(self._self2process_queue,self._process2self_queue,self._stopsignal_queue))
        return process
    def _FrameStream(self,self2process_queue,process2self_queue,stopsignal_queue):
        def _InitCurSaveDir(cur_save_dir):
            if(os.path.exists(cur_save_dir)==False):
                os.mkdir(cur_save_dir)
            if(os.path.exists(cur_save_dir+"/img")==False):
                os.mkdir(cur_save_dir+"/img")
            else:
                all_files=os.listdir(cur_save_dir+"/img")
                for _file in all_files:
                    file_path=cur_save_dir+"/img/"+_file
                    if(os.path.isfile(file_path)==True):
                        os.remove(file_path)
            if(os.path.exists(cur_save_dir+"/json")==False):
                os.mkdir(cur_save_dir+"/json")
            else:
                all_files=os.listdir(cur_save_dir+"/json")
                for _file in all_files:
                    file_path=cur_save_dir+"/json/"+_file
                    if(os.path.isfile(file_path)==True):
                        os.remove(file_path)

        init_dict=self2process_queue.get()
        camera_idx=init_dict["CAMERA_IDX"]
        video_hw=init_dict["VIDEO_HW"]
        show_bool=init_dict["SHOW_BOOL"]
        save_dir=init_dict.get("SAVE_DIR","")

        cur_bboxes=None

        camera_capture=cv2.VideoCapture(camera_idx)
        if not camera_capture.isOpened():
            raise Exception("CameraStream _FrameStream Error: Cannot open camera.")
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH,video_hw[1])
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,video_hw[0])

        data_dict={}
        recording_bool=False
        recording_count=0
        frame_count=0
        cur_save_dir=None
        while(1):
            if(stopsignal_queue.empty()==False):
                signal_dict=stopsignal_queue.get()
                if(signal_dict.get("STOP_SIGNAL",False)==True):
                    stopsignal_queue.put({"STOP_SIGNAL":True})
                    break
    
            ret,frame=camera_capture.read()
            show_frame=frame.copy()
            while not process2self_queue.empty():
                process2self_queue.get()
            process2self_queue.put({"FRAME":frame})

            if(show_bool==True):
                if(recording_bool==True):
                    show_frame=DrawStatus(show_frame,status_str="Recording",bgr=(0,255,0))
                    cv2.imwrite(cur_save_dir+"/img/"+str(frame_count)+".jpg",frame)
                else:
                    show_frame=DrawStatus(show_frame,status_str="Not Recording",bgr=(0,0,255))

                if(self2process_queue.empty()==False):
                    data_dict=self2process_queue.get()
                bboxes=data_dict.get("BBOXES",None)
                if(type(bboxes)!=type(None)):
                    cur_bboxes=bboxes
                    if(recording_bool==True):Bboxes2JSON(bboxes,cur_save_dir+"/json/"+str(frame_count)+".json")
                    bboxes=None
                if(type(cur_bboxes)!=type(None)):
                    show_frame=Drawing(show_frame,cur_bboxes,thickness=1)
                cv2.imshow('DEMO',show_frame)

                key_type=cv2.waitKey(1)
                if(key_type==ord("q")):
                    stopsignal_queue.put({"STOP_SIGNAL":True})
                    break
                elif(key_type==ord('r')):
                    if(recording_bool==False):
                        cur_save_dir=save_dir+"/"+str(recording_count)
                        _InitCurSaveDir(cur_save_dir)
                        frame_count=0
                        recording_count+=1
                        recording_bool=True
                elif(key_type==ord('c')):
                    if(recording_bool==True):
                        recording_bool=False
            frame_count+=1
            time.sleep(0.001)
        if(show_bool==True):cv2.destroyAllWindows()
        camera_capture.release()
        return
    def Start(self):
        self._process.start()
        self.GetFrame()
        return
    def Stop(self):
        self._stopsignal_queue.put({"STOP_SIGNAL":True})
        self._process.terminate()
        self._process.join()
        return
    def StopChecking(self):
        if(self._stopsignal_queue.empty()==True):
            return False
        else:
            signal_dict=self._stopsignal_queue.get()
            return signal_dict.get("STOP_SIGNAL",False)
    def GetFrame(self):
        frame=self._process2self_queue.get()["FRAME"]
        self._process2self_queue.put({"FRAME":frame})
        return frame
    def UpdateBboxes(self,bboxes):
        self._self2process_queue.put({"BBOXES":bboxes})
        return 
    def IsAlive(self):
        return self._process.is_alive()
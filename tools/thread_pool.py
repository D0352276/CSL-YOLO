import threading
from threading import Thread
import queue
import time

class ThreadPool:
    _lock = threading.RLock()
    def __init__(self,thread_num):
        self._funtion_queue=queue.Queue()
        self._result_dict={}
        self._thread=[Thread(target=self._Thread,args=(i,)) for i in range(thread_num)]
        self._thread_switch=[0 for i in range(thread_num)]
        self._thread_processing_signal=[0 for i in range(thread_num)]
        self._thread_num=thread_num
        self._execute_count=0
    def _Thread(self,thread_number):
        this_thread_number=thread_number
        self._thread_processing_signal[this_thread_number]=1
        while(self._thread_switch[this_thread_number]==1):
            if(self._funtion_queue.empty()==False):
                execute_mark,function,parameter=self._funtion_queue.get()
                if(parameter==None):result=function()
                else:result=function(parameter)
                self._result_dict[execute_mark]=result
            time.sleep(0.1)
        self._thread_processing_signal[this_thread_number]=0
        return 
    def __call__(self,function,parameter=None):
        return self.Push(function,parameter)
    def __getitem__(self,execute_mark):
        return self.GetResult(execute_mark)
    def Push(self,function,parameter=None):
        with ThreadPool._lock:
            execute_mark=self._execute_count
            self._funtion_queue.put([execute_mark,function,parameter])
            self._execute_count+=1
        return execute_mark
    def GetResult(self,execute_mark,block=True,time_gap=0.1):
        if(block==True):
            while(1):
                result=self._result_dict.get(execute_mark,False)
                if(result!=False):break
                time.sleep(time_gap)
        else:
            result=self._result_dict.get(execute_mark,False)
        if(result!=False):del self._result_dict[execute_mark]
        return result
    def Start(self):
        for i in range(len(self._thread_switch)):
            self._thread_switch[i]=1
            self._thread[i].start()
    def Stop(self,waiting_time_out=0):
        for i in range(len(self._thread_switch)):
            self._thread_switch[i]=0
        start_time=time.time()
        while(time.time()-start_time<=waiting_time_out):
            if(sum(self._thread_processing_signal)==0):
                return True
        return False

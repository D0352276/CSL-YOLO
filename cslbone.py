import tensorflow as tf
from modules import ConvBN,CSLModule

class CSLMBlock(tf.Module):
    def __init__(self,filters,t,down_rate=1.0,blck_len=1,use_se=True,name="cslmblck"):
        super(CSLMBlock,self).__init__(name=name)
        self._filters=filters
        self._t=t
        self._down_rate=down_rate
        self._blck_len=blck_len
        self._use_se=use_se
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._cslm_list=[]
        self._first_cslm=CSLModule(self._filters,t=self._t,down_rate=self._down_rate,use_se=self._use_se,name=self._name+"_first_cslm")
        for i in range(self._blck_len-1):
            self._cslm_list.append(CSLModule(self._filters,t=self._t,name=self._name+"_cslm_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        first_x=self._first_cslm(input_ts)
        x=first_x
        for i in range(self._blck_len-1):
            x=self._cslm_list[i](x)
        if(self._blck_len>1):
            output_ts=first_x+x
        else:
            output_ts=first_x
        return output_ts

class CSLBone(tf.Module):
    def __init__(self,name="cslbone"):
        super(CSLBone,self).__init__(name=name)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._convbn=ConvBN(16,(3,3),(2,2),name=self._name+"_convbn")
        self._cslmblck_1=CSLMBlock(filters=16,t=3,down_rate=1,blck_len=2,name=self._name+"_cslmblck_1")
        self._cslmblck_2=CSLMBlock(filters=32,t=3,down_rate=0.5,blck_len=2,name=self._name+"_cslmblck_2")
        self._cslmblck_3=CSLMBlock(filters=64,t=3,down_rate=0.5,blck_len=4,name=self._name+"_cslmblck_3")
        self._cslmblck_4=CSLMBlock(filters=128,t=3,down_rate=0.5,blck_len=6,name=self._name+"_cslmblck_4")
        self._cslmblck_5=CSLMBlock(filters=192,t=3,down_rate=1,blck_len=6,name=self._name+"_cslmblck_5")
        self._cslmblck_6=CSLMBlock(filters=256,t=3,down_rate=0.5,blck_len=8,name=self._name+"_cslmblck_6")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._convbn(input_ts)
        x=self._cslmblck_1(x)
        x=self._cslmblck_2(x)
        x1=self._cslmblck_3(x)
        x=self._cslmblck_4(x1)
        x2=self._cslmblck_5(x)
        x3=self._cslmblck_6(x2)
        return x1,x2,x3

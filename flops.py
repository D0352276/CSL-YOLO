import math
import numpy as np

class TensorOP:
    def __init__(self,input_shape,flops=0):
        self._cur_flops=flops
        self._cur_ts_shape=input_shape
    def Conv(self,filters,kernal_size=(3,3),strides=(1,1)):
        self._cur_ts_shape[0]=round(self._cur_ts_shape[0]/strides[0])
        self._cur_ts_shape[1]=round(self._cur_ts_shape[1]/strides[0])
        flops=self._cur_ts_shape[0]*self._cur_ts_shape[1]*self._cur_ts_shape[2]*kernal_size[0]**2*filters
        self._cur_ts_shape[2]=filters
        self._cur_flops+=flops
        return flops
    def DWConv(self,kernal_size=(3,3),strides=(1,1)):
        self._cur_ts_shape[0]=round(self._cur_ts_shape[0]/strides[0])
        self._cur_ts_shape[1]=round(self._cur_ts_shape[1]/strides[0])
        flops=self._cur_ts_shape[0]*self._cur_ts_shape[1]*self._cur_ts_shape[2]*kernal_size[0]**2
        self._cur_flops+=flops
        return flops
    def SeparableConv(self,filters,kernal_size=(3,3),strides=(1,1)):
        flops=self.DWConv(kernal_size,strides)
        _flops=self.Conv(filters,(1,1),(1,1))
        return flops+_flops
    def MaxPooling(self):
        self._cur_ts_shape[0]=round(self._cur_ts_shape[0]/2)
        self._cur_ts_shape[1]=round(self._cur_ts_shape[1]/2)
        return 0
    def AvgPooling(self):
        return self.MaxPooling()
    def AdaptAvgPooling(self,output_hw):
        self._cur_ts_shape[0]=output_hw[0]
        self._cur_ts_shape[1]=output_hw[1]
        return 
    def GlobalAvgPool(self):
        self._cur_ts_shape[0]=1
        self._cur_ts_shape[1]=1
        return 0
    def Upsample(self):
        self._cur_ts_shape[0]=self._cur_ts_shape[0]*2
        self._cur_ts_shape[1]=self._cur_ts_shape[1]*2
        return 0
    def Resize(self,output_hw):
        return self.AdaptAvgPooling(output_hw)
    def Concat(self,tsop):
        tsop_shape=tsop.Shape()
        if(self._cur_ts_shape[0]!=tsop_shape[0] or self._cur_ts_shape[1]!=tsop_shape[1]):
            raise Exception("TensorOP Concat Error: Shape not match.")
        self._cur_ts_shape[2]+=tsop_shape[2]
        self._cur_flops+=tsop.Flops()
        return 0
    def Add(self,tsop):
        tsop_shape=tsop.Shape()
        if(self._cur_ts_shape[0]!=tsop_shape[0] or self._cur_ts_shape[1]!=tsop_shape[1] or self._cur_ts_shape[2]!=tsop_shape[2]):
            raise Exception("TensorOP Add Error: Shape not match.")
        self._cur_flops+=tsop.Flops()
        return 0
    def Multiply(self,tsop):
        tsop_shape=tsop.Shape()
        if(self._cur_ts_shape[0]!=tsop_shape[0] or self._cur_ts_shape[1]!=tsop_shape[1] or self._cur_ts_shape[2]!=tsop_shape[2]):
            raise Exception("TensorOP Add Error: Shape not match.")
        self._cur_flops+=tsop.Flops()
        return 0
    def Flops(self):
        return self._cur_flops
    def Shape(self):
        return self._cur_ts_shape.copy()
    def Copy(self,include_flops=True):
        if(include_flops==True):
            return TensorOP(self.Shape(),self.Flops())
        else:
            return TensorOP(self.Shape())

def SELayer(in_tsop):
    in_shape=in_tsop.Shape()
    tsop=in_tsop.Copy(include_flops=False)
    tsop.GlobalAvgPool()
    tsop.Conv(in_shape[2]//4,kernal_size=(1,1),strides=(1,1))
    tsop.Conv(in_shape[2],kernal_size=(1,1),strides=(1,1))
    tsop.Resize(in_shape[0:2])
    tsop.Multiply(in_tsop)
    return tsop

def CSPGModule(in_tsop,filters,t,down_rate=1.0,use_se=False):
    in_shape=in_tsop.Shape()
    out_hw=[round(in_shape[0]*down_rate),round(in_shape[1]*down_rate)]

    out_ch_1=filters//2
    out_ch_2=filters-out_ch_1

    skip_tsop=in_tsop.Copy(include_flops=False)
    if(down_rate<1.0):
        skip_tsop.AdaptAvgPooling(out_hw)
    skip_tsop.Conv(out_ch_1,kernal_size=(1,1),strides=(1,1))

    skip_expand_tsop=None
    for i in range(t):
        buf=skip_tsop.Copy(include_flops=False)
        buf.DWConv(kernal_size=(3,3),strides=(1,1))
        if(skip_expand_tsop==None):
            skip_expand_tsop=buf
        else:
            skip_expand_tsop.Concat(buf)
    
    out_tsop=in_tsop.Copy(include_flops=True)
    out_tsop.DWConv(kernal_size=(3,3),strides=(1,1))
    if(down_rate<1.0):
        out_tsop.AdaptAvgPooling(out_hw)

    out_tsop.Concat(skip_expand_tsop)
    out_tsop.DWConv(kernal_size=(3,3),strides=(1,1))
    
    if(use_se==True):
        out_tsop=SELayer(out_tsop)
    out_tsop.Conv(out_ch_2,kernal_size=(1,1),strides=(1,1))
    out_tsop.Concat(skip_tsop)
    return out_tsop

def CSLBlock(in_tsop,filters,t,down_rate=1.0,blck_len=1,use_se=True):
    x=CSPGModule(in_tsop,filters,t=t,down_rate=down_rate,use_se=use_se)
    for i in range(blck_len-1):
        x=CSPGModule(x,filters,t=t,down_rate=1.0,use_se=False)
    out_tsop=x
    return out_tsop

def CSLBone(in_tsop):
    in_tsop.Conv(16,(3,3),(2,2))
    x=in_tsop
    x=CSLBlock(x,16,3,1.0,2)
    x=CSLBlock(x,32,3,0.5,2)
    x=CSLBlock(x,64,3,0.5,4)
    x=CSLBlock(x,128,3,0.5,6)
    x=CSLBlock(x,192,3,1.0,6)
    x=CSLBlock(x,256,3,0.5,8)
    out_tsop=x
    return out_tsop

def InputBIFusion(btm_tsop,top_tsop):
    btm_tsop=btm_tsop.Copy(False)
    btm_shape=btm_tsop.Shape()
    top_tsop=top_tsop.Copy(False)
    top_shape=top_tsop.Shape()

    target_hw=[round((btm_shape[0]+top_shape[0])/2),round((btm_shape[1]+top_shape[1])/2)]

    btm_tsop.AdaptAvgPooling(target_hw)
    top_tsop.Resize(target_hw)
    out_tsop=top_tsop.Copy(include_flops=True)

    out_tsop.Add(btm_tsop)
    out_tsop=CSPGModule(out_tsop,round((btm_shape[2]+top_shape[2])/2),t=2,use_se=True)
    return out_tsop

def ConstraintPhase1(in_tsop_list):
    l1,l2,l3,l4,l5=in_tsop_list

    _l1=l1.Copy(False)
    _l3=l3.Copy(False)
    l2_hw=l2.Shape()[:2]
    _l1.AdaptAvgPooling(l2_hw)
    _l3.Resize(l2_hw)
    l2.Add(_l1)
    l2.Add(_l3)
    l2=CSPGModule(l2,l2.Shape()[2],t=2,use_se=True)

    _l3=l3.Copy(False)
    _l5=l5.Copy(False)
    l4_hw=l4.Shape()[:2]
    _l3.AdaptAvgPooling(l4_hw)
    _l5.Resize(l4_hw)
    l4.Add(_l3)
    l4.Add(_l5)
    l4=CSPGModule(l4,l4.Shape()[2],t=2,use_se=True)

    return l1,l2,l3,l4,l5

def ConstraintPhase2(in_tsop_list):
    l1,l2,l3,l4,l5=in_tsop_list

    _l2_up=l2.Copy(False)
    _l2_down=l2.Copy(False)
    _l2_up.Resize(l1.Shape()[:2])
    l1.Add(_l2_up)
    l1=CSPGModule(l1,l1.Shape()[2],t=2,use_se=True)
    _l2_down.AdaptAvgPooling(l3.Shape()[:2])
    l3.Add(_l2_down)

    _l4_up=l4.Copy(False)
    _l4_down=l4.Copy(False)
    _l4_up.Resize(l3.Shape()[:2])
    l3.Add(_l4_up)
    l3=CSPGModule(l3,l3.Shape()[2],t=2,use_se=True)

    _l4_down.AdaptAvgPooling(l5.Shape()[:2])
    l5.Add(_l4_down)
    l5=CSPGModule(l5,l5.Shape()[2],t=2,use_se=True)

    return l1,l2,l3,l4,l5

def ConstraintFPN(in_tsop_list,repet=2):
    for i in range(repet):
        in_tsop_list=ConstraintPhase1(in_tsop_list)
        in_tsop_list=ConstraintPhase2(in_tsop_list)
    out_tsop_list=in_tsop_list
    return out_tsop_list
    

input_wh=320
t=3
anchors_len=3
labels_len=80

fpn_filters=112
fpn_repeat=3

in_tsop=TensorOP([input_wh,input_wh,3])
out_tsop=CSLBone(in_tsop)
cslbone_backbone_flops=out_tsop.Flops()
print(cslbone_backbone_flops/1024**2)

l1_tsop=TensorOP([input_wh//8,input_wh//8,40])
l3_tsop=TensorOP([input_wh//16,input_wh//16,112])
l5_tsop=TensorOP([input_wh//32,input_wh//32,192])

l1_tsop=CSPGModule(l1_tsop,fpn_filters,t,down_rate=1.0,use_se=True)
l3_tsop=CSPGModule(l3_tsop,fpn_filters,t,down_rate=1.0,use_se=True)
l5_tsop=CSPGModule(l5_tsop,fpn_filters,t,down_rate=1.0,use_se=True)
l2_tsop=InputBIFusion(l1_tsop,l3_tsop)
l4_tsop=InputBIFusion(l3_tsop,l5_tsop)

print(l1_tsop.Shape())
print(l2_tsop.Shape())
print(l3_tsop.Shape())
print(l4_tsop.Shape())
print(l5_tsop.Shape())

in_tsop_list=[l1_tsop,l2_tsop,l3_tsop,l4_tsop,l5_tsop]
out_tsop_list=ConstraintFPN(in_tsop_list,fpn_repeat)

l1_tsop,l2_tsop,l3_tsop,l4_tsop,l5_tsop=out_tsop_list

# l1_tsop.Conv(fpn_filters,(3,3),(1,1))
# l2_tsop.Conv(fpn_filters,(3,3),(1,1))
# l3_tsop.Conv(fpn_filters,(3,3),(1,1))
# l4_tsop.Conv(fpn_filters,(3,3),(1,1))
# l5_tsop.Conv(fpn_filters,(3,3),(1,1))

l1_tsop.Conv(anchors_len*(labels_len+5),(1,1),(1,1))
l2_tsop.Conv(anchors_len*(labels_len+5),(1,1),(1,1))
l3_tsop.Conv(anchors_len*(labels_len+5),(1,1),(1,1))
l4_tsop.Conv(anchors_len*(labels_len+5),(1,1),(1,1))
l5_tsop.Conv(anchors_len*(labels_len+5),(1,1),(1,1))

b0_backbone_flops=1759510528
m2_backbone_flops=499*1024*1024*1
head_flops=l1_tsop.Flops()+l2_tsop.Flops()+l3_tsop.Flops()+l4_tsop.Flops()+l5_tsop.Flops()

tot_flops=cslbone_backbone_flops+head_flops
print(tot_flops/1024**2)
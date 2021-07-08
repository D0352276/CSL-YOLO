import math

class Operation:
    def __init__(self):
        self._in_shape=None
        self._out_shape=None
        self._flops=0
    def __call__(self,in_shape):
        self._in_shape=self._CopyShape(in_shape)
        self._out_shape,self._flops=self._Operation(self._CopyShape(in_shape))
        return self._out_shape
    def _CopyShape(self,shape):
        return [shape[0],shape[1],shape[2]]
    def _Operation(self,in_shape):
        out_shape=in_shape
        flops=0
        return out_shape,flops
    def InShape(self):
        if(self._in_shape==None):raise Exception("Operation Error: The OP has not been used.")
        return self._CopyShape(self._in_shape)
    def OutShape(self):
        if(self._out_shape==None):raise Exception("Operation Error: The OP has not been used.")
        return self._CopyShape(self._out_shape)
    def FLOPs(self):
        return self._flops

class Conv(Operation):
    def __init__(self,filters,kernal_size=(3,3),strides=(1,1)):
        super(Conv,self).__init__()
        self._filters=filters
        self._kernal_size=kernal_size
        self._strides=strides
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=round(out_shape[0]/self._strides[0])
        out_shape[1]=round(out_shape[1]/self._strides[1])
        flops=out_shape[0]*out_shape[1]*in_shape[2]*self._kernal_size[0]**2*self._filters
        out_shape[2]=self._filters
        return out_shape,flops

class DWConv(Operation):
    def __init__(self,kernal_size=(3,3),strides=(1,1)):
        super(DWConv,self).__init__()
        self._kernal_size=kernal_size
        self._strides=strides
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=round(out_shape[0]/self._strides[0])
        out_shape[1]=round(out_shape[1]/self._strides[1])
        flops=out_shape[0]*out_shape[1]*in_shape[2]*self._kernal_size[0]**2
        return out_shape,flops
 
class MaxPooling(Operation):
    def __init__(self,strides=(2,2)):
        super(MaxPooling,self).__init__()
        self._strides=strides
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=round(in_shape[0]/self._strides[0])
        out_shape[1]=round(in_shape[1]/self._strides[1])
        flops=0
        return out_shape,flops

class AvgPooling(MaxPooling):pass

class AdaptAvgPooling(Operation):
    def __init__(self,output_hw):
        super(AdaptAvgPooling,self).__init__()
        self._output_hw=output_hw
    def _Operation(self,in_shape):
        if(self._output_hw[0]>in_shape[0] or self._output_hw[1]>in_shape[1]):
            raise Exception("AdaptAvgPooling Error: The 'output_hw' must be small than 'in_shape'.")
        out_shape=in_shape
        out_shape[0]=round(self._output_hw[0])
        out_shape[1]=round(self._output_hw[1])
        flops=0
        return out_shape,flops

class GlobalAvgPooling(Operation):
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=1
        out_shape[1]=1
        flops=0
        return out_shape,flops

class Upsample(Operation):
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=round(out_shape[0]*2)
        out_shape[1]=round(out_shape[1]*2)
        flops=0
        return out_shape,flops

class Resize(Operation):
    def __init__(self,output_hw):
        super(Resize,self).__init__()
        self._output_hw=output_hw
    def _Operation(self,in_shape):
        out_shape=in_shape
        out_shape[0]=round(self._output_hw[0])
        out_shape[1]=round(self._output_hw[1])
        flops=0
        return out_shape,flops



class Module:
    def __init__(self):
        self._in_shapes=None
        self._out_shapes=None
        self._flops=0
    def __call__(self,in_shapes):
        self._Build(self._CopyShapes(in_shapes))
        self._in_shapes=self._CopyShapes(in_shapes)
        self._out_shapes=self._Operation(self._CopyShapes(in_shapes))
        self._flops=self._FLOPsCalculator(self)
        return self._out_shapes
    def _Build(self,in_shapes):
        return
    def _CopyShapes(self,shapes):
        return [[shape[0],shape[1],shape[2]] for shape in shapes]
    def _FLOPsCalculator(self,module):
        flops=0
        if(issubclass(type(module),Operation)==True):
            return module.FLOPs()
        elif(type(module)==list):
            for sub_module in module:
                flops+=self._FLOPsCalculator(sub_module)
        elif(issubclass(type(module),Module)==True):
            for key in module.__dict__.keys():
                sub_module=module.__dict__[key]
                flops+=self._FLOPsCalculator(sub_module)
        else:return 0
        return flops
    def _Operation(self,in_shapes):
        out_shapes=in_shapes
        return out_shapes
    def InShapes(self):
        if(self._in_shapes==None):raise Exception("Operation Error: The Module has not been used.")
        return self._CopyShapes(self._in_shapes)
    def OutShape(self):
        if(self._out_shapes==None):raise Exception("Operation Error: The Module has not been used.")
        return self._CopyShapes(self._out_shapes)
    def FLOPs(self):
        return self._flops

class Concat(Module):
    def _Operation(self,in_shapes):
        out_shape=[in_shapes[0][0],in_shapes[0][1],0]
        last_shape=None
        for in_shape in in_shapes:
            if(last_shape!=None and last_shape[:2]!=in_shape[:2]):
                raise Exception("Concat Error: The width and height of 'in_shapes' must be equal.")
            out_shape[2]+=in_shape[2]
            last_shape=in_shape
        return [out_shape]

class Add(Module):
    def _Operation(self,in_shapes):
        out_shape=[in_shapes[0][0],in_shapes[0][1],in_shapes[0][2]]
        last_shape=None
        for in_shape in in_shapes:
            if(last_shape!=None and last_shape!=in_shape):
                raise Exception("Concat Error: The width and height and channel of 'in_shapes' must be equal.")
            last_shape=in_shape
        return [out_shape]

class Multiply(Module):
    def _Operation(self,in_shapes):
        in_shapes_len=len(in_shapes)
        if(in_shapes_len!=2):
            raise Exception("Multiply Error: The length of 'in_shapes' must be 2.")
        if(in_shapes[0]!=in_shapes[1]):
            raise Exception("Multiply Error: The width and height and channel of 'in_shapes' must be equal.")
        out_shape=in_shapes[0]
        return [out_shape]

class SEModule(Module):
    def _Build(self,in_shapes):
        in_shape=in_shapes[0]
        self._global_pool=GlobalAvgPooling()
        self._conv1=Conv(in_shape[2]//4)
        self._conv2=Conv(in_shape[2])
        self._resize=Resize(in_shape[:2])
        self._multiply=Multiply()
    def _Operation(self,in_shapes):
        in_shape=in_shapes[0]
        x=self._global_pool(in_shape)
        x=self._conv1(x)
        x=self._conv2(x)
        x=self._resize(x)
        out_shape=self._multiply([x,in_shape])[0]
        return [out_shape]

class CSLModule(Module):
    def __init__(self,filters,t=2,down_rate=1,use_se=False):
        super(CSLModule,self).__init__()
        self._filters=filters
        self._t=t
        self._down_rate=down_rate
        self._use_se=use_se
    def _Build(self,in_shapes):
        in_shape=in_shapes[0]
        self._p1_ch=round(self._filters/2)
        self._p2_ch=self._filters-self._p1_ch
        self._out_shape=in_shape
        self._out_shape[0]=math.ceil(self._out_shape[0]*self._down_rate)
        self._out_shape[1]=math.ceil(self._out_shape[1]*self._down_rate)
        self._out_shape[2]=self._filters

        if(self._down_rate<1):
            self._skip_pool=AdaptAvgPooling(self._out_shape[:2])
        self._skip_conv=Conv(self._p1_ch,kernal_size=(1,1))

        self._skip_expands=[]
        for i in range(self._t):
            self._skip_expands.append(DWConv())
        self._skip_concat=Concat()

        self._input_expand=DWConv()
        if(self._down_rate<1.0):
            self._input_pool=AdaptAvgPooling(self._out_shape[:2])
        self._exapnd_concat=Concat()

        self._expand_dwconv=DWConv()
        if(self._use_se==True):
            self._sem=SEModule()
        self._expand_compress=Conv(filters=self._p2_ch,kernal_size=(1,1))
        self._out_concat=Concat()
    def _Operation(self,in_shapes):
        in_shape=in_shapes[0]
        if(self._down_rate<1):
            p1=self._skip_pool(in_shape)
            p1=self._skip_conv(p1)
        else:
            p1=self._skip_conv(in_shape)

        p1_expands=[]
        for i in range(self._t):
            p1_expands.append(self._skip_expands[i](p1))
        p1_expands=self._skip_concat(p1_expands)[0]
        in_expand=self._input_expand(in_shape)
        if(self._down_rate<1.0):
            in_expand=self._input_pool(in_expand)
        p2=self._exapnd_concat([p1_expands,in_expand])[0]
        p2=self._expand_dwconv(p2)
        if(self._use_se==True):
            p2=self._sem([p2])[0]
        p2=self._expand_compress(p2)
        out_shape=self._out_concat([p1,p2])[0]
        return [out_shape]

class CSLBlock(Module):
    def __init__(self,filters,t,down_rate=1.0,blck_len=1,use_se=True):
        super(CSLBlock,self).__init__()
        self._filters=filters
        self._t=t
        self._down_rate=down_rate
        self._blck_len=blck_len
        self._use_se=use_se
    def _Build(self,in_shapes):
        self._cslms=[CSLModule(self._filters,self._t,down_rate=self._down_rate,use_se=self._use_se)]
        for i in range(self._blck_len-1):
            self._cslms.append(CSLModule(self._filters,self._t,down_rate=1.0,use_se=False))
    def _Operation(self,in_shapes):
        x=in_shapes
        for i in range(self._blck_len):
            x=self._cslms[i](x)
        out_shapes=x
        return out_shapes

class CSLBone(Module):
    def _Build(self,input_shapes):
        self._conv=Conv(16,(3,3),(2,2))
        self._cslblck_1=CSLBlock(16,3,1.0,2)
        self._cslblck_2=CSLBlock(32,3,0.5,2)
        self._cslblck_3=CSLBlock(64,3,0.5,4)
        self._cslblck_4=CSLBlock(128,3,0.5,6)
        self._cslblck_5=CSLBlock(192,3,1.0,6)
        self._cslblck_6=CSLBlock(256,3,0.5,8)
    def _Operation(self,input_shapes):
        x=self._conv(input_shapes[0])
        x=self._cslblck_1([x])
        x=self._cslblck_2(x)
        x=self._cslblck_3(x)
        x=self._cslblck_4(x)
        x=self._cslblck_5(x)
        x=self._cslblck_6(x)
        out_shapes=x
        return out_shapes

class InputBIFusion(Module):
    def _Build(self,input_shapes):
        btm_shape,top_shape=input_shapes
        output_shape=[round((btm_shape[0]+top_shape[0])/2),round((btm_shape[1]+top_shape[1])/2),round((btm_shape[2]+top_shape[2])/2)]
        self._adapt_pool=AdaptAvgPooling(output_shape[:2])
        self._resize=Resize(output_shape[:2])
        self._add=Add()
        self._cslm=CSLModule(output_shape[2])
    def _Operation(self,input_shapes):
        btm_shape,top_shape=input_shapes
        x1=self._adapt_pool(btm_shape)
        x2=self._resize(top_shape)
        out_shapes=self._add([x1,x2])
        out_shapes=self._cslm(out_shapes)
        return out_shapes
        
class FusionPhase1(Module):
    def _Build(self,input_shapes):
        l1_shape,l2_shape,l3_shape,l4_shape,l5_shape=input_shapes
        self._l1_down=AdaptAvgPooling(l2_shape[:2])
        self._l3_up=Resize(l2_shape[:2])
    def _Operation(self,input_shapes):
        l1_shape,l2_shape,l3_shape,l4_shape,l5_shape=input_shapes


in_shape=[224,224,3]
cslbone=CSLBone()
out_shape=cslbone([in_shape])[0]
print(out_shape)
print(cslbone.FLOPs()/1024**2)

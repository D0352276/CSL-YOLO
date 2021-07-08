import tensorflow as tf
import numpy as np
swish=tf.keras.layers.Lambda(lambda x:x*tf.math.sigmoid(x))
hard_sigmoid=tf.keras.layers.Lambda(lambda x:tf.nn.relu6(x+3.0)/6.0)
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))

class ConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="convbn"):
        super(ConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class DepthConvBN(tf.Module):
    def __init__(self,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="depthconvbn"):
        super(DepthConvBN,self).__init__(name=name)
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding=self._padding,
                                                        use_bias=self._bias,
                                                        name=self._name+"_depthconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._depthconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class SeparableConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="spbconvbn"):
        super(SeparableConvBN,self).__init__(name=name)
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._spbconv=tf.keras.layers.SeparableConv2D(self._filters,
                                                      self._kernel_size,
                                                      self._strides,
                                                      depth_multiplier=1,
                                                      padding=self._padding,
                                                      use_bias=self._bias,
                                                      name=self._name+"_spbconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._spbconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class SEModule(tf.Module):
    def __init__(self,name="sem"):
        super(SEModule,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._conv1=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=tf.nn.relu,name=self._name+"_conv1")
        self._conv2=ConvBN(input_ch,kernel_size=(1,1),use_bn=False,activation=hard_sigmoid,name=self._name+"_conv2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._gap(input_ts)
        x=tf.reshape(x,[-1,1,1,input_ch])
        x=self._conv1(x)
        x=self._conv2(x)
        output_ts=input_ts*x
        return output_ts

class AdaptAvgPooling(tf.Module):
    def __init__(self,output_hw,name="adaptavgpooling"):
        super(AdaptAvgPooling,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_hw):
        stride=np.floor((input_hw/(self._output_hw)))
        pool_size=input_hw-(self._output_hw-1)*stride
        self._avgpool=tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                       strides=stride,
                                                       name=self._name+"_avgpool")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_hw=input_ts.get_shape().as_list()[1:3]
        self._Build(input_hw)
        output_ts=self._avgpool(input_ts)
        return output_ts

class AdaptUpsample(tf.Module):
    def __init__(self,output_hw,name="adaptupsample"):
        super(AdaptUpsample,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class CSLModule(tf.Module):
    def __init__(self,filters,t=2,down_rate=1,use_se=False,activation=mish,name="cslmodule"):
        super(CSLModule,self).__init__(name=name)
        self._filters=filters
        self._t=round(t)
        self._down_rate=down_rate
        self._use_se=use_se
        self._activation=activation
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ts):
        input_shape=input_ts.get_shape().as_list()
        out_hw=np.array([np.ceil(input_shape[1]*self._down_rate),np.ceil(input_shape[2]*self._down_rate)])

        out_ch_1=self._filters//2
        out_ch_2=self._filters-out_ch_1

        #skip connect
        if(self._down_rate<1.0):
            self._skip_pool=AdaptAvgPooling(out_hw,name=self._name+"_skip_pool")
        self._skip_conv=ConvBN(out_ch_1,kernel_size=(1,1),use_bn=True,activation=None,name=self._name+"_skip_conv")

        #expand
        self._skip_expands=[]
        for i in range(self._t):
            dconv=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_skip_expand_"+str(i))
            self._skip_expands.append(dconv)
        self._input_expand=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_input_expand")
        if(self._down_rate<1.0):
            self._input_pool=AdaptAvgPooling(out_hw,name=self._name+"_input_pool")
        self._expand_bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_expand_bn")
        self._expand_act=tf.keras.layers.Activation(self._activation,name=self._name+"_expand_act")

        #extract
        self._depth_conv=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_depth_conv")
        self._extract_bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_extract_bn")
        if(self._use_se==True):
            self._sem=SEModule(name=self._name+"_sem")
        self._extract_act=tf.keras.layers.Activation(self._activation,name=self._name+"_extract_act")

        #compress
        self._compress=ConvBN(out_ch_2,kernel_size=(1,1),use_bn=True,activation=None,name=self._name+"_compress")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        self._Build(input_ts)
        
        x=input_ts
        if(self._down_rate<1.0):
            x=self._skip_pool(input_ts)
        p1=self._skip_conv(x)

        skip_expand_x=[]
        for i in range(self._t):
            skip_x=self._skip_expands[i](p1)
            skip_expand_x.append(skip_x)
        skip_expand_x=tf.concat(skip_expand_x,axis=-1)
        input_expand_x=self._input_expand(input_ts)
        if(self._down_rate<1.0):
            input_expand_x=self._input_pool(input_expand_x)
        x=tf.concat([input_expand_x,skip_expand_x],axis=-1)
        x=self._expand_bn(x)
        x=self._expand_act(x)

        x=self._depth_conv(x)
        x=self._extract_bn(x)
        if(self._use_se==True):
            x=self._sem(x)
        x=self._extract_act(x)
        
        p2=self._compress(x)

        output_ts=tf.concat([p1,p2],axis=-1)
        return output_ts

class InputBIFusion(tf.Module):
    def __init__(self,name="inputbufusion"):
        super(InputBIFusion,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,btm_shape,top_shape):
        btm_shape=np.array(btm_shape)
        top_shape=np.array(top_shape)
        target_shape=np.round((btm_shape+top_shape)/2)
        self._btm_down=AdaptAvgPooling(target_shape[0:2],name=self._name+"_btm_down")
        self._top_up=AdaptUpsample(target_shape[0:2],name=self._name+"_top_up")
        self._cslm=CSLModule(filters=top_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_cslm")
    @tf.Module.with_name_scope
    def __call__(self,btm_ts,top_ts):
        btm_shape=btm_ts.get_shape().as_list()[1:]
        top_shape=top_ts.get_shape().as_list()[1:]
        self._Build(btm_shape,top_shape)
        btm_down=self._btm_down(btm_ts)
        top_up=self._top_up(top_ts)
        x=btm_down+top_up
        output_ts=self._cslm(x)
        return output_ts

class _FusionPhase1(tf.Module):
    def __init__(self,name="fusionphase1"):
        super(FusionPhase1,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l2_shape,l4_shape):
        l2_shape=np.array(l2_shape)
        l4_shape=np.array(l4_shape)
        self._l1_down=AdaptAvgPooling(l2_shape[0:2],name=self._name+"_l1_down")
        self._l3_up=AdaptUpsample(l2_shape[0:2],name=self._name+"_l3_up")
        self._l2_cslm=CSLModule(filters=l2_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l2_cslm")
        self._l3_down=AdaptAvgPooling(l4_shape[0:2],name=self._name+"_l3_down")
        self._l5_up=AdaptUpsample(l4_shape[0:2],name=self._name+"_l5_up")
        self._l4_cslm=CSLModule(filters=l4_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l4_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l2_shape=l2.get_shape().as_list()[1:]
        l4_shape=l4.get_shape().as_list()[1:]
        self._Build(l2_shape,l4_shape)

        l1_down=self._l1_down(l1)
        l3_up=self._l3_up(l3)
        
        l2=l2+l1_down+l3_up
        l2=self._l2_cslm(l2)

        l3_down=self._l3_down(l3)
        l5_up=self._l5_up(l5)
        l4=l4+l3_down+l5_up
        l4=self._l4_cslm(l4)
        return [l1,l2,l3,l4,l5]

class FusionPhase1(tf.Module):
    def __init__(self,name="fusionphase1"):
        super(FusionPhase1,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l2_shape,l4_shape):
        l2_shape=np.array(l2_shape)
        l4_shape=np.array(l4_shape)
        self._l1_to_l2=AdaptAvgPooling(l2_shape[0:2],name=self._name+"_l1_to_l2")
        self._l1_to_l4=AdaptAvgPooling(l4_shape[0:2],name=self._name+"_l1_to_l4")
        self._l3_to_l2=AdaptUpsample(l2_shape[0:2],name=self._name+"_l3_to_l2")
        self._l3_to_l4=AdaptAvgPooling(l4_shape[0:2],name=self._name+"_l3_to_l4")
        self._l5_to_l2=AdaptUpsample(l2_shape[0:2],name=self._name+"_l5_to_l2")
        self._l5_to_l4=AdaptUpsample(l4_shape[0:2],name=self._name+"_l5_to_l4")

        self._l2_cslm=CSLModule(filters=l2_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l2_cslm")
        self._l4_cslm=CSLModule(filters=l4_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l4_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l2_shape=l2.get_shape().as_list()[1:]
        l4_shape=l4.get_shape().as_list()[1:]
        self._Build(l2_shape,l4_shape)
        
        l2=l2+self._l1_to_l2(l1)+self._l3_to_l2(l3)+self._l5_to_l2(l5)
        l2=self._l2_cslm(l2)

        l4=l4+self._l1_to_l4(l1)+self._l3_to_l4(l3)+self._l5_to_l4(l5)
        l4=self._l4_cslm(l4)
        return [l1,l2,l3,l4,l5]

class _FusionPhase2(tf.Module):
    def __init__(self,name="fusionphase2"):
        super(FusionPhase2,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l1_shape,l3_shape,l5_shape):
        l1_shape=np.array(l1_shape)
        l3_shape=np.array(l3_shape)
        l5_shape=np.array(l5_shape)
        self._l2_up=AdaptUpsample(l1_shape[0:2],name=self._name+"_l2_up")
        self._l1_cslm=CSLModule(filters=l1_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l1_cslm")

        self._l2_down=AdaptAvgPooling(l3_shape[0:2],name=self._name+"_l2_down")
        self._l4_up=AdaptUpsample(l3_shape[0:2],name=self._name+"_l4_up")
        self._l3_cslm=CSLModule(filters=l3_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l3_cslm")

        self._l4_down=AdaptAvgPooling(l5_shape[0:2],name=self._name+"_l4_down")
        self._l5_cslm=CSLModule(filters=l5_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l5_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l1_shape=l1.get_shape().as_list()[1:]
        l3_shape=l3.get_shape().as_list()[1:]
        l5_shape=l5.get_shape().as_list()[1:]
        self._Build(l1_shape,l3_shape,l5_shape)

        l2_up=self._l2_up(l2)
        l1=l1+l2_up
        l1=self._l1_cslm(l1)

        l2_down=self._l2_down(l2)
        l4_up=self._l4_up(l4)
        l3=l3+l2_down+l4_up
        l3=self._l3_cslm(l3)

        l4_down=self._l4_down(l4)
        l5=l5+l4_down
        l5=self._l5_cslm(l5)
        return [l1,l2,l3,l4,l5]

class FusionPhase2(tf.Module):
    def __init__(self,name="fusionphase2"):
        super(FusionPhase2,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l1_shape,l3_shape,l5_shape):
        l1_shape=np.array(l1_shape)
        l3_shape=np.array(l3_shape)
        l5_shape=np.array(l5_shape)

        self._l2_to_l1=AdaptUpsample(l1_shape[0:2],name=self._name+"_l2_to_l1")
        self._l2_to_l3=AdaptAvgPooling(l3_shape[0:2],name=self._name+"_l2_to_l3")
        self._l2_to_l5=AdaptAvgPooling(l5_shape[0:2],name=self._name+"_l2_to_l5")

        self._l4_to_l1=AdaptUpsample(l1_shape[0:2],name=self._name+"_l4_to_l1")
        self._l4_to_l3=AdaptUpsample(l3_shape[0:2],name=self._name+"_l4_to_l3")
        self._l4_to_l5=AdaptAvgPooling(l5_shape[0:2],name=self._name+"_l4_to_l5")

    
        self._l1_cslm=CSLModule(filters=l1_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l1_cslm")
        self._l3_cslm=CSLModule(filters=l3_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l3_cslm")
        self._l5_cslm=CSLModule(filters=l5_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l5_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l1_shape=l1.get_shape().as_list()[1:]
        l3_shape=l3.get_shape().as_list()[1:]
        l5_shape=l5.get_shape().as_list()[1:]
        self._Build(l1_shape,l3_shape,l5_shape)

        l1=l1+self._l2_to_l1(l2)+self._l4_to_l1(l4)
        l1=self._l1_cslm(l1)

        l3=l3+self._l2_to_l3(l2)+self._l4_to_l3(l4)
        l3=self._l3_cslm(l3)

        l5=l5+self._l2_to_l5(l2)+self._l4_to_l5(l4)
        l5=self._l5_cslm(l5)
        return [l1,l2,l3,l4,l5]

class CSLFPN(tf.Module):
    def __init__(self,repeat=3,name="cslfpn"):
        super(CSLFPN,self).__init__(name=name)
        self._repeat=repeat
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._fusion_phase1_list=[]
        self._fusion_phase2_list=[]
        for i in range(self._repeat):
            self._fusion_phase1_list.append(FusionPhase1(name=self._name+"_phase1_"+str(i)))
            self._fusion_phase2_list.append(FusionPhase2(name=self._name+"_phase2_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        out_ts_list=input_ts_list
        for i in range(self._repeat):
            last_out_ts_list=out_ts_list.copy()
            out_ts_list=self._fusion_phase1_list[i](out_ts_list)
            out_ts_list=self._fusion_phase2_list[i](out_ts_list)
            for ts_idx in range(len(out_ts_list)):
                out_ts_list[ts_idx]=out_ts_list[ts_idx]+last_out_ts_list[ts_idx]
        return out_ts_list

class VanillaFPN(tf.Module):
    def __init__(self,name="vanillafPN"):
        super(VanillaFPN,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l1_shape,l2_shape,l3_shape,l4_shape,l5_shape):
        l1_shape=np.array(l1_shape)
        l2_shape=np.array(l2_shape)
        l3_shape=np.array(l3_shape)
        l4_shape=np.array(l4_shape)
        l5_shape=np.array(l5_shape)
        self._l2_up=AdaptUpsample(l1_shape[0:2],name=self._name+"_l2_up")
        self._l3_up=AdaptUpsample(l2_shape[0:2],name=self._name+"_l3_up")
        self._l4_up=AdaptUpsample(l3_shape[0:2],name=self._name+"_l4_up")
        self._l5_up=AdaptUpsample(l4_shape[0:2],name=self._name+"_l5_up")
        self._l1_conv=ConvBN(filters=l1_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l1_conv")
        self._l2_conv=ConvBN(filters=l2_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l2_conv")
        self._l3_conv=ConvBN(filters=l3_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l3_conv")
        self._l4_conv=ConvBN(filters=l4_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l4_conv")
        self._l5_conv=ConvBN(filters=l5_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l5_conv")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l1_shape=l1.get_shape().as_list()[1:]
        l2_shape=l2.get_shape().as_list()[1:]
        l3_shape=l3.get_shape().as_list()[1:]
        l4_shape=l4.get_shape().as_list()[1:]
        l5_shape=l5.get_shape().as_list()[1:]
        self._Build(l1_shape,l2_shape,l3_shape,l4_shape,l5_shape)

        l4=l4+self._l5_up(l5)
        l3=l3+self._l4_up(l4)
        l2=l2+self._l3_up(l3)
        l1=l1+self._l2_up(l2)

        l1=self._l1_conv(l1)
        l2=self._l2_conv(l2)
        l3=self._l3_conv(l3)
        l4=self._l4_conv(l4)
        l5=self._l5_conv(l5)
        out_ts_list=[l1,l2,l3,l4,l5]
        return out_ts_list

class InvertedResidual(tf.Module):
    def __init__(self,filters,t,kernel_size=(3,3),strides=(1,1),first_layer=False,name="invertedresidual"):
        super(InvertedResidual,self).__init__(name=name)
        self._filters=filters
        self._t=t
        self._kernel_size=kernel_size
        self._strides=strides
        self._first_layer=first_layer
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_channel):
        if(self._first_layer==True):input_channel=self._filters
        tchannel=int(input_channel*self._t)
        if(self._first_layer==True):
            self._convbn_1=ConvBN(tchannel,self._kernel_size,(2,2),name=self._name+"_convbn_1")
        else:
            self._convbn_1=ConvBN(tchannel,(1,1),(1,1),name=self._name+"_convbn_1")
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding="same",
                                                        use_bias=False,
                                                        name=self._name+"_depthconv")
        self._bn=tf.keras.layers.BatchNormalization(name=self._name+"_bn")
        self._relu6=tf.keras.layers.ReLU(max_value=6.0,name=self._name+"_relu")
        self._convbn_2=ConvBN(self._filters,(1,1),(1,1),use_relu=False,name=self._name+"_convbn_2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._convbn_1(input_ts)
        x=self._depthconv(x)
        x=self._bn(x)
        x=self._relu6(x)
        x=self._convbn_2(x)
        if(self._strides==(1,1) and self._filters==input_ch):
            x=(input_ts+x)
        output_ts=x
        return output_ts
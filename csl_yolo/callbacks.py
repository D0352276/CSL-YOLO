import tensorflow as tf
import os
import math

class LearningRateReducer(tf.keras.callbacks.Callback):
    def __init__(self,lr_tune_dict={}):
        super(LearningRateReducer,self).__init__()
        self._lr_tune_dict=lr_tune_dict
    def on_epoch_end(self,epoch,logs={}):
        lr_tune=self._lr_tune_dict.get(epoch,False)
        if(lr_tune!=False):
            self.model.optimizer.lr.assign(lr_tune)
        return 

class Stabilizer(tf.keras.callbacks.Callback):
    def __init__(self,security_boundary=0.1):
        super(Stabilizer,self).__init__()
        self._security_boundary=1+security_boundary
        self._last_loss=None
    def on_train_begin(self,logs={}):
        if(os.path.isfile("stabilizer.hdf5")==True):
            os.remove("stabilizer.hdf5")
        self.model.save_weights("stabilizer.hdf5")
    def on_train_end(self,logs={}):
        os.remove("stabilizer.hdf5")
    def on_epoch_end(self,epoch,logs={}):
        loss=logs.get('loss')
        if(math.isnan(loss)==True):
            for var in self.model.optimizer.variables():
                var.assign(tf.zeros_like(var))
            self.model.load_weights("stabilizer.hdf5")
        elif(self._last_loss==None or loss<self._last_loss*self._security_boundary):
            self.model.save_weights("stabilizer.hdf5")
            self._last_loss=loss

class WeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self,save_path):
        super(WeightsSaver,self).__init__()
        self._save_path=save_path
    def on_epoch_begin(self,epoch,logs={}):
        self.model.save_weights(self._save_path)
        return 
        
class BestWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self,save_path,eval_function,eval_parms=None,init_metric=0.0):
        super(BestWeightsSaver,self).__init__()
        self._save_path=save_path
        self._eval_function=eval_function
        self._eval_parms=eval_parms
        self._cur_metric=init_metric
    def on_epoch_begin(self,epoch,logs={}):
        if(self._eval_parms==None or self._eval_parms==[]):
            metric=self._eval_function(self.model)
        else:
            metric=self._eval_function(self.model,*self._eval_parms)
        if(metric<self._cur_metric):return
        if(metric>self._cur_metric):
            self._cur_metric=metric
            self.model.save_weights(self._save_path)
        return 
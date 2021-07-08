def FreezeLayers(model,keys=None,freeze_type="LAYERS"):
    if(freeze_type=="ALL"):
        for layer in model.layers:
            layer.trainable=False
    elif(keys==None):
        raise Exception("FreezeLayers Error: The arg 'keys' can't be None.")
    elif(type(keys[0])==str and freeze_type=="LAYERS"):
        for name in keys:
            model.get_layer(name=name).trainable=False
    elif(type(keys[0])==int and freeze_type=="LAYERS"):
        for idx in keys:
            model.get_layer(index=idx).trainable=False
    return 
def UnfreezeLayers(model,keys=None,freeze_type="LAYERS"):
    if(freeze_type=="ALL"):
        for layer in model.layers:
            layer.trainable=True
    elif(keys==None):
        raise Exception("FreezeLayers Error: The arg 'keys' can't be None.")
    elif(type(keys[0])==str and freeze_type=="LAYERS"):
        for name in keys:
            model.get_layer(name=name).trainable=True
    elif(type(keys[0])==int and freeze_type=="LAYERS"):
        for idx in keys:
            model.get_layer(index=idx).trainable=True
    return 
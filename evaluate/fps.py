import datetime
import numpy as np
def FramePerSecond(model,input_shape,test_num=1000):
    imgs=[]
    for i in range(test_num):
        print("Creating........."+str(i)+"th test_img")
        imgs.append(np.array([np.zeros(input_shape)]))
    tot_img=len(imgs)
    start=datetime.datetime.now()
    for img in imgs:
        model.predict_on_batch(img)
    end=datetime.datetime.now()
    cost_seconds=(end-start).seconds
    return tot_img/cost_seconds
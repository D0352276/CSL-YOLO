def IOU(box_1,box_2):
    x1,y1,w1,h1=box_1
    half_w1=w1/2
    half_h1=h1/2

    x2,y2,w2,h2=box_2
    half_w2=w2/2
    half_h2=h2/2

    max_x=max(x1+half_w1,x2+half_w2)
    min_x=min(x1-half_w1,x2-half_w2)
    inter_w=w1+w2-(max_x-min_x)

    max_y=max(y1+half_h1,y2+half_h2)
    min_y=min(y1-half_h1,y2-half_h2)
    inter_h=h1+h2-(max_y-min_y)

    if(inter_w<=0 or inter_h<=0):
        iou=0 
    else:
        inter_area=inter_w*inter_h
        area1=w1*h1
        area2=w2*h2
        union_area=area1+area2-inter_area
        iou=inter_area/union_area
    return iou
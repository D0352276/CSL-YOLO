def ParsingList(list_str):
    if(list_str[0]!="[" or list_str[-1]!="]"):
        return list_str
    list_str=list_str[1:]
    list_str=list_str[:-1]
    val_list_buf=list_str.split(",")
    val_list=[]
    for i in range(len(val_list_buf)-1):
        if(val_list_buf[i][0]=="[" and val_list_buf[i+1][-1]=="]"):
            val_list.append(val_list_buf[i]+","+val_list_buf[i+1])
        elif(val_list_buf[i][0]!="[" and val_list_buf[i][-1]!="]"):
            val_list.append(val_list_buf[i])
    if(val_list_buf[-1][-1]!="]"):val_list.append(val_list_buf[-1])
    for i,elemt in enumerate(val_list):
        val_list[i]=ParsingList(elemt)
    return val_list

def ParsingValue(val):
    if(type(val)==list):
        for i,elemt in enumerate(val):
            elemt=ParsingValue(elemt)
            val[i]=elemt
    else:
        try:val=float(val)
        except:
            if(val=="True"):val=True
            elif(val=="False"):val=False
            pass
    return val

def ParsingCfg(cfg_path):
    cfg_dict={}
    fin=open(cfg_path,"r")
    lines=fin.read().split("\n")
    fin.close()
    for line in lines:
        if(line=="" or line[0]=="#"):continue
        key,val=line.split("=")
        val=ParsingList(val)
        val=ParsingValue(val)
        cfg_dict[key]=val
    return cfg_dict
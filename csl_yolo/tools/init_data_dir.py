import os

def InitDataDir(data_dir):
    imgs_dir=data_dir
    jsons_dir=imgs_dir+"/json"
    if(os.path.exists(imgs_dir)==False):
        os.mkdir(imgs_dir)
    if(os.path.exists(jsons_dir)==False):
        os.mkdir(jsons_dir)
    jsons_file=os.listdir(jsons_dir)
    for json_file in jsons_file:
        file_path=jsons_dir+"/"+json_file
        if(os.path.isfile(file_path)==True):
            os.remove(file_path)
    imgs_file=os.listdir(imgs_dir)
    for img_file in imgs_file:
        file_path=imgs_dir+"/"+img_file
        if(os.path.isfile(file_path)==True):
            os.remove(file_path)
    return
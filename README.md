# CSL-YOLO: A New Lightweight Object Detection System for Edge Computing

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project provides a SOTA level lightweight YOLO called CSL-YOLO,

it is achieving better detection performance with only 43% FLOPs and 52% parameters than Tiny-YOLOv4.

<div align=center>
<img src=https://github.com/D0352276/CSL-YOLO/blob/main/demo/result_img.png width=60% />
</div>

## Requirements

- [Tensorflow2.0](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/)
- [Python 3](https://www.python.org/)


## How to Get Started?

### Predict
```bash
python3 main.py -p cfg/predict_coco.cfg
```

### Train
```bash
python3 main.py -t cfg/train_coco.cfg
```

### Eval on MS-COCO
```bash
python3 main.py -ce cfg/eval_coco.cfg
```

## More Info
<img src=https://github.com/D0352276/CSL-YOLO/blob/main/demo/result_table.png width=100% />

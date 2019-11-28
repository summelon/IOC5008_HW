# HW3

## Introduction
    - Save your training & test images in `data` folder 
    - Run `makeTxt.py` to produce label annotation .txt file
    - Run `SVHN_train.py` to train your model. 
    - Run `SVHN_detect.py` to test FPS or output .json prediction file.

### `SVHN_train.py` command line argument:
```python=
parser.add_argument('--epochs', type=int, default=273) # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--transfer', action='store_true', help='transfer learning')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--weights', type=str, default='weights/ultralytics49.pt', help='initial weights')
parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--var', type=float, help='debug variable')
```

### `SVHN_detect.py` command line argument:
```python=
parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg file path')
parser.add_argument('--data', type=str, default='data/SVHN.data', help='coco.data file path')
parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file') # input file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='test/', help='source')
parser.add_argument('--output', type=str, default='predict_result/', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-json', action='store_true', help='choose to save predicted .json file')
parser.add_argument('--inf-detail', action='store_true', help='choose to print inference detail')
```

```
.
├── cfg/
│   ├── README.md
│   ├── yolov3.cfg
│   ├── yolov3-spp.cfg
│   └── yolov3-tiny.cfg
├── data/
│   ├── images/
│   │   └── README.md
│   ├── labels/
│   │   └── README.md
│   ├── README.md
│   ├── SVHN.data
│   ├── SVHN.names
│   ├── SVHN_train.txt
│   └── SVHN_val.txt
├── makeTxt.py
├── models.py
├── predict_result/
│   ├── mAP_0.40088_0856154_3.json
│   └── README.md
├── README.md
├── SVHN_detect.py
├── SVHN_test.py
├── SVHN_train.py
├── utils/
│   ├── adabound.py
│   ├── datasets.py
│   ├── gcp.sh
│   ├── google_utils.py
│   ├── parse_config.py
│   ├── README.md
│   ├── torch_utils.py
│   └── utils.py
└── weights/
    ├── download_yolov3_weights.sh
    └── README.md
```

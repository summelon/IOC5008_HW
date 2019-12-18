# HW4

## Environment preparation

1. Clone this repository
2. Install dependencies on your host or docker container
  ```bash
  pip3 install -r requirements.txt
  ```
3. Run setup from the repository root directory
  ```bash
  python3 setup.py install 
  ```
  
4. Edit `Cococonfig` class in `train.py` according to your images size in dataset
![](https://i.imgur.com/spXIDWE.png)


5. Run `train.py` to train/evaluate your model 
  - train your model
  ```bash
  python train.py train --model imagenet --dataset pascal
  ```
  - Evaluate your trained weights
  ```bash
  python train.py evaluate --model logs/trained_time/your_weights.h5 --dataset pascal
  ```

## Appendix

- Command line argument in `train.py`
  ```python
  parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate' on MS COCO")
  parser.add_argument('--dataset', required=True, metavar="/path/to/coco/", help='Directory of the MS-COCO dataset')
  parser.add_argument('--year', required=False, default=DEFAULT_DATASET_YEAR, metavar="<year>", help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
  parser.add_argument('--model', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
  parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
  parser.add_argument('--limit', required=False, default=500, metavar="<image count>", help='Images to use for evaluation (default=500)')
  parser.add_argument('--download', required=False, default=False, metavar="<True|False>", help='Automatically download and unzip MS-COCO files (default=False)', type=bool)

  ```

- Directory tree
```shell=
├── logs
│   └── README.md
├── mrcnn
│   ├── config.py
│   ├── __init__.py
│   ├── model.py
│   ├── parallel_model.py
│   ├── README.md
│   ├── utils.py
│   └── visualize.py
├── pascal
│   ├── eval
│   │   └── README.md
│   ├── eval.json
│   ├── train
│   │   └── README.md
│   ├── train.json
│   └── val
│       └── README.md
├── README.md
├── requirements.txt
├── setup.py
└── train.py

```

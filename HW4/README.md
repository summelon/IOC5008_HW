# HW4

# Environment preparation

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
  ```bash
  
  ```

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

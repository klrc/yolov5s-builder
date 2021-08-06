### Requirements
- pycocotools
- tqdm
- coco dataset at `../datasets/coco`, structured as:
coco/train2017
coco/val2017
coco/labels/train2017
coco/labels/val2017
coco/train2017.txt
coco/val2017.txt

### Usage

#### step1: customized dataset
1. edit rules at `scripts/create_dataset.py:line 97`
2. edit config file at `data/coco-x.yaml` 
3. run `python scripts/create_dataset.py`


#### step2: train model
1. run `wandb offline` to turn off syncing.
2. `python scripts/train.py --data coco-x.yaml --img 416`

#### step3: evaluation
`python scripts/val.py --weights runs/train/exp4/weights/best.pt --data coco-x.yaml --img 416`

#### step4: export model
`python scripts/export.py -i runs/train/exp4/weights/best.pt -o build/yolov5s.onnx`
`make all`

#### clean dir
to clear all build & training history, run `make clean`
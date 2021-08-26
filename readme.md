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

#### build caffe model
`make all`

#### clean dir
to clear all build & training history, run `make clean`
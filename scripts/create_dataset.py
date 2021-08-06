import os
import shutil
from tqdm import tqdm
import multiprocessing

from pycocotools.coco import COCO

# Store annotations and train2014/ val2014/... in this folder
source_root = '../datasets/coco'

# the path you want to save your results for coco to voc
target_root = '../datasets/coco-x'
img_dir = target_root + '/images'
anno_dir = target_root + '/labels'
datasets_list = ['train2017', 'val2017']
classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck', 'chair', 'tv', 'cell phone', 'cat', 'dog', 'horse']
true_cid_dict = {
    'car': 2,
    'bicycle': 1,
    'person': 0,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'chair': 56,
    'tv': 62,
    'cell phone': 67,
    'cat': 17,
    'dog': 18,
    'horse': 19,
}

# create directory
try:
    shutil.rmtree(target_root)
except Exception:
    pass
os.mkdir(target_root)
os.mkdir(img_dir)
os.mkdir(anno_dir)
for dataset in datasets_list:
    os.mkdir(f'{img_dir}/{dataset}')
    os.mkdir(f'{anno_dir}/{dataset}')


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def get_objects(coco, img, classes, cls_id):
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    devide_w, devide_h = 1./img['width'], 1./img['height']
    objs = []
    for ann in anns:
        cid = ann['category_id']
        class_name = classes[cid]
        if class_name in classes_names:
            if 'bbox' in ann:
                bbox = ann['bbox']
                obj = dict(
                    cid=true_cid_dict[class_name],
                    class_name=class_name,
                    x=(bbox[0] + bbox[2]/2.0 - 1)*devide_w,
                    y=(bbox[1] + bbox[3]/2.0 - 1)*devide_h,
                    w=bbox[2]*devide_w,
                    h=bbox[3]*devide_h,
                )
                objs.append(obj)
    return objs


def save_image(coco, dataset, file_name):
    source_path = f'{source_root}/{dataset}/{file_name}'
    target_path = f'{img_dir}/{dataset}/{file_name}'
    shutil.copy(source_path, target_path)
    with open(f'{target_root}/{dataset}.txt', 'a') as f:
        f.write(f'./images/{dataset}/{file_name}\n')


def save_label(coco, dataset, file_name, objs):
    target_path = f'{anno_dir}/{dataset}/{file_name.replace(".jpg",".txt")}'
    with open(target_path, 'w') as f:
        for obj in objs:
            line = (obj['cid'], obj['x'], obj['y'], obj['w'], obj['h'])
            f.write(" ".join([str(x) for x in line]) + '\n')


def worker(tasks):
    for dataset, img_id, coco, classes, classes_ids in tqdm(tasks):
        img = coco.loadImgs(img_id)[0]
        file_name = img['file_name']
        objs = get_objects(coco, img, classes, classes_ids)

        # rules to extract data here -------------------------
        ratio_label_person = len([x for x in objs if x['class_name'] == 'person'])/len(objs)
        ratio_label_chair = len([x for x in objs if x['class_name'] == 'chair'])/len(objs)
        if ratio_label_person < 0.8 and ratio_label_chair < 0.4:  # skip samples for balance
            save_image(coco, dataset, file_name)
            save_label(coco, dataset, file_name, objs)


def run():
    for dataset in datasets_list:
        coco = COCO(f'{source_root}/annotations/instances_{dataset}.json')
        classes = id2name(coco)
        classes_ids = coco.getCatIds(catNms=classes_names)

        for cls in classes_names:
            tasks = []
            cls_id = coco.getCatIds(catNms=[cls])
            img_ids = coco.getImgIds(catIds=cls_id)
            for img_id in img_ids:
                tasks.append((dataset, img_id, coco, classes, classes_ids))
            p = multiprocessing.Process(target=worker, args=(tasks,))
            p.start()


if __name__ == "__main__":
    run()

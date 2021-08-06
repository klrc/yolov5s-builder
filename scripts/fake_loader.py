import torch
from model.yolov5s import Xyolov5s


def attempt_create(pretrained=True):
    model = Xyolov5s(pretrained=pretrained)
    model.as_relu()
    return model


def attempt_load(weights, device):
    model = attempt_create(pretrained=False)
    if weights.endswith('.pt'):
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['ema' if ckpt.get('ema') else 'model'].state_dict())
    return model.float().fuse().eval().to(device)

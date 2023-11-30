import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', autoshape=False, pretrained=False)

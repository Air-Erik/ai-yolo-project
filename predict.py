from ultralytics import YOLO
from PIL import Image
import cv2
import torch

# Загрузка модели для пердсказания
model = YOLO("runs/detect/train3/weights/last.pt")

results = model(['test2.jpg'])

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save('result.jpg')
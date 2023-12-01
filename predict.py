from ultralytics import YOLO
from PIL import Image

# Загрузка модели для пердсказания
model = YOLO("runs/detect/train/weights/best.pt")

results = model(['test3.jpg'])

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save('result.jpg')
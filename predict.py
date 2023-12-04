from ultralytics import YOLO
from PIL import Image

# Загрузка модели для пердсказания
model = YOLO("runs/detect/train4/weights/best.pt")

results = model(['test.jpg'])

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save('result.jpg')
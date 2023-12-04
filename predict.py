from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Загрузка модели для пердсказания
model = YOLO("runs/detect/train3/weights/best.pt")

results = model(['test/test1.jpg', 'test/test2.jpg'])

i = 0

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(f'result/result{i}.jpg')
    
    frames = r.boxes.xyxy.cpu().numpy()
    percent = r.boxes.conf.cpu().numpy()
    clas = r.boxes.cls.cpu().numpy()
    
    df = pd.DataFrame(frames, columns=['x1', 'y1', 'x2', 'y2'])
    df['percent'], df['class'] = pd.DataFrame(percent), pd.DataFrame(clas)
    df.to_csv(f'result/database{i}.csv')
    
    print(df.head())
    i += 1
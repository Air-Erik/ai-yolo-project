from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os
import glob

# Название папки с обученными весами (название используется для вывода результата)
custom_weights = 'train6'
# путь к папке с изображениями для тестирования
pth_test = 'test/AutoCAD_Topo_v7/'


# Загрузка модели для пердсказания
model = YOLO(f"runs/detect/{custom_weights}/weights/best.pt")

# Список тестовых изображений
source = []
for file in glob.glob(pth_test + '*'):
    source.append(file)
print(source)

# Предсказание. Параметр conf определяет достоверный порог вероятности при котором засчитывается обнаружение
results = model(source, conf=0.65)

# Счетчик
i = 0

# Последовательная обработка результатов по каждому изображению
for r in results:
    # Сохранение картинки с рамками захвата
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(f'result/{custom_weights}/result{i}.jpg')
    
    # Конвертация результатов из tensor в numpy
    frames = r.boxes.xyxy.cpu().numpy()
    percent = r.boxes.conf.cpu().numpy()
    clas = r.boxes.cls.cpu().numpy()
    
    # Создание таблицы pandas с последующей выгрузкой в CSV
    df = pd.DataFrame(frames, columns=['x1', 'y1', 'x2', 'y2'])
    df['percent'], df['class'] = pd.DataFrame(percent), pd.DataFrame(clas)
    df.to_csv(f'result/{custom_weights}/data_{i}.csv')
    
    # Печать таблицы
    print(df.head())
    i += 1 # Увеличение счетчика
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Загрузка модели для пердсказания
custom_model = 'train6'
model = YOLO(f"runs/detect/{custom_model}/weights/best.pt")

source = ['test/AutoCAD_Topo_v7/test_1_0_0.jpg', 'test/AutoCAD_Topo_v7/test_1_0_1.jpg',
          'test/AutoCAD_Topo_v7/test_1_0_0x2.jpg', 'test/AutoCAD_Topo_v7/test_1_0_0x1.5.jpg',
          'test/AutoCAD_Topo_v7/test_1_0_0x2(1280).jpg']

# Предсказание. Параметр conf определяет достоверный порог вероятности при котором засчитывается обнаружение
results = model(source, conf=0.65)

# Счетчик
i = 0

# Последовательная обработка результатов по каждому изображению
for r in results:
    # Сохранение картинки с рамками захвата
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(f'result/{custom_model}/result{i}.jpg')
    
    # Конвертация результатов из tensor в numpy
    frames = r.boxes.xyxy.cpu().numpy()
    percent = r.boxes.conf.cpu().numpy()
    clas = r.boxes.cls.cpu().numpy()
    
    # Создание таблицы pandas с последующей выгрузкой в CSV
    df = pd.DataFrame(frames, columns=['x1', 'y1', 'x2', 'y2'])
    df['percent'], df['class'] = pd.DataFrame(percent), pd.DataFrame(clas)
    df.to_csv(f'result/{custom_model}/database{i}.csv')
    
    # Печать таблицы
    print(df.head())
    i += 1 # Увеличение счетчика
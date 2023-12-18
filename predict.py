from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os
import psycopg
from psycopg import sql
import yaml


# Название папки с обученными весами (название используется для вывода
# результата)
custom_weights = 'train6'
# Путь к папке с изображениями для классификации
pth_raw = 'test/AutoCAD_Topo_v7/'
# Путь к папке с датасетом, для чтения названий классов
pth_dataset = 'datasets/AutoCAD_Topo_v7/data.yaml'

# Загрузка модели для пердсказания
model = YOLO(f"runs/detect/{custom_weights}/weights/best.pt")

# Создает список с путями ко всем файлам в папке pth_raw
# Также создает список имен файлов без расширений. Он используется для вывода
# результата
source = []
file_names = []
for dirpath, dirnames, filenames in os.walk(pth_raw):
    for filename in filenames:
        source.append(os.path.join(dirpath, filename))
        file_names.append(os.path.splitext(filename)[0])

# Извлекает из датасета имена для классов
with open(pth_dataset) as yaml_file:
    yaml_read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
# Извлекает из словаря только имена классов
class_names = yaml_read_data['names']

# Предсказание. Параметр conf определяет достоверный порог вероятности при
# котором засчитывается обнаружение
results = model(source, conf=0.70)

# Счетчик нужен для обработки нескольких изображений
i = 0

# Создание директории для вывода результатов
if not os.path.exists(f'result/{custom_weights}'):
    os.mkdir(f'result/{custom_weights}')

# Подготвока SQL запроса по которому будет создавать таблица
query = sql.SQL('''
    CREATE TABLE {} (
        x_1 numeric,
        y_1 numeric,
        x_2 numeric,
        y_2 numeric,
        percent numeric,
        file_name text,
        class text,
        class_id numeric)
    ''').format(
        sql.Identifier(pth_raw)
        )

# Создание таблицы в базе данных для записи результатов работы программы
# по подготовленому ранее запросу
with psycopg.connect('dbname=ai_database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:
        try:
            cur.execute(query)
        except:
            pass

# Последовательная обработка результатов по каждому изображению
for r in results:
    # Сохранение картинки с рамками захвата
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(f'result/{custom_weights}/{file_names[i]}.jpg')

    # Конвертация результатов из tensor в numpy
    frames = r.boxes.xyxy.cpu().numpy()
    percent = r.boxes.conf.cpu().numpy()
    clas = r.boxes.cls.cpu().numpy()

    # Создание таблицы pandas с последующей выгрузкой в CSV
    df = pd.DataFrame(frames, columns=['x1', 'y1', 'x2', 'y2'])
    df['percent'], df['class_id'] = pd.DataFrame(percent), pd.DataFrame(clas)
    # Приведение к типу float64 потому что postgreSQL ругается на тип
    # данных float32
    df = df.astype('float64')
    # Заполняет название классов
    for j in df['class_id']:
        df.loc[df['class_id'] == j, 'class'] = class_names[int(j)]

    # Запись в базу данных
    with psycopg.connect('dbname=ai_database user=ayrapetov_es \
    password=1111') as conn:

        # Создание элемента курсора
        with conn.cursor() as cur:
            # Перебор строк дата фрейма в цикле
            for index, row in df.iterrows():
                # Отправка SQL запроса
                cur.execute(sql.SQL('''
                    INSERT INTO {} (
                        x_1,
                        y_1,
                        x_2,
                        y_2,
                        percent,
                        file_name,
                        class,
                        class_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ''').format(
                        sql.Identifier(pth_raw)),
                            (row['x1'],
                             row['y1'],
                             row['x2'],
                             row['y2'],
                             row['percent'],
                             file_names[i],
                             row['class'],
                             row['class_id'])
                    )
            cur.close()

        conn.commit()

    i += 1  # Увеличение счетчика

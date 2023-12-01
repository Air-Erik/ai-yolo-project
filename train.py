from ultralytics import YOLO
from clearml import Task

task = Task.init(project_name="my project", task_name="my task")

# Объявление модели
model = YOLO("yolov5s.yaml")

# Обучение модели на наборе данных
result = model.train(data='datasets/test_1v/data.yaml', epochs=100, imgsz=640)

# Оценка качества тренировки модели на проверочном сете
result = model.val()
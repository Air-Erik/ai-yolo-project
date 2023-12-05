from clearml import Task
from ultralytics import YOLO

def main():
    # ClearML; Создание объекта задачи для clearml, описывает проект и название текущей сессии
    task = Task.init(
        project_name="my_project",
        task_name="yolov8l"
    )
    
    # ClearML; Определение модели на которой будет происходить обучение
    model_variant = "yolov8l"
    task.set_parameter("model_variant", model_variant)
    
    # YOLO; Объявление модели
    model = YOLO(f'{model_variant}.pt')
    
    # ClearML; Устанавка параметров обучения, передаются в функцию обучения
    args = dict(data='datasets/test_1v/data.yaml', epochs=100, imgsz=640, freeze=10)
    task.connect(args)
    
    # YOLO; Обучение модели на наборе данных
    result = model.train(**args)

# Необходимое условие для многопоточности. Без данной конструкции не возможен параллелизм
if __name__ == '__main__':
    main()
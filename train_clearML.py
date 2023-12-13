from clearml import Task
from ultralytics import YOLO
import torch


def main():
    # Проверка работы CUDA и вычисления на GPU
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    # ClearML; Создание объекта задачи для clearml, описывает проект и
    # название текущей сессии
    task = Task.init(
        project_name="AutoCAD",
        task_name="yolov8l"
    )

    # ClearML; Определение модели на которой будет происходить обучение
    model_variant = "yolov8l"
    task.set_parameter("model_variant", model_variant)

    # YOLO; Объявление модели
    model = YOLO(f'{model_variant}.pt')

    # ClearML; Устанавка параметров обучения, передаются в функцию обучения
    args = dict(data='datasets/AutoCAD_Topo_v7/data.yaml',
                epochs=300,
                imgsz=640,
                freeze=10)
    task.connect(args)

    # YOLO; Обучение модели на наборе данных
    result = model.train(**args)


# Необходимое условие для многопоточности. Без данной конструкции не возможен
# параллелизм
if __name__ == '__main__':
    main()

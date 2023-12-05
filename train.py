from ultralytics import YOLO
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

def main():
    # Comet;Определение класса эксперемент 
    experiment = Experiment(
        api_key="k6KiCLTqcdnM42KrPJ0pWbqBY",
        project_name="yolo",
        workspace="air-erik"
    )
    
    # Comet; Опеределение гиперпараметров дял модели
    hyper_params = {
        "learning_rate": 0.5,
        "steps": 100000,
        "batch_size": 50,
    }
    experiment.log_parameters(hyper_params)
    
    # YOLO; Объявление модели
    model = YOLO("yolov5n.pt")
    
    # YOLO; Обучение модели на наборе данных
    result = model.train(data='datasets/test_1v/data.yaml', epochs=3, imgsz=640, freeze=10)
    
    # Comet; Протоколирование модели
    log_model(experiment, model, model_name="TheModel")

# Необходимое условие для многопоточности. Без данной конструкции не возможен параллелизм
if __name__ == '__main__':
    main()
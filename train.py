import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/yg/zhaoziteng/ultralytics-main-zzt/yolov8l-HSFPN4-mamba.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/yg/ZS/ultralytics-main-6-19/NWPU VHR-10.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=12,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
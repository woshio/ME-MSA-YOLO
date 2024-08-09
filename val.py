import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/yg/zhaoziteng/ultralytics-main-zzt/weight/Visdrone/ME-MAS-YOLO.pt')
    model.val(data='/home/yg/ZS/ultralytics-main-6-19/VisDrone.yaml',
              split='test',
              imgsz=800,
              batch=10,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
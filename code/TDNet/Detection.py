import os
import sys
import cv2
import numpy as np
import torch
from TDNet import Utils
from TDNet import Calibration


class YOLOv5:
    def __init__(self,
                root,  # YOLOv5的根目录，用于导入模块和加载权重文件
                weights=None,  # 模型权重文件的路径，默认为None，会使用root目录下的'yolov5l.pt'
                imgsz=640,  # 推理时的图像大小（像素）
                device='',  # 指定运行设备，如'0'表示使用第一个CUDA设备，'cpu'表示使用CPU
                half=True,  # 是否使用FP16半精度进行推理，默认为True
                dnn=True,  # 是否使用OpenCV DNN进行ONNX推理，默认为True
                ):
        # 导入所需的模块和函数
        sys.path.insert(1, root)
        import torch
        import torch.backends.cudnn as cudnn
        self.torch = torch
        
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_boxes, check_requirements
        from utils.torch_utils import select_device
        self.letterbox = letterbox
        self.check_img_size = check_img_size
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_boxes
        self.check_requirements = check_requirements

        # 加载模型权重
        if weights is None: weights = os.path.join(root, 'yolov5l.pt')
        print('\n <<< Model is running on {} >>> \n'.format('CUDA GPU' if torch.cuda.is_available() else 'CPU'))
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights=weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # 检查图像大小
        self.half = half & (self.pt or jit or engine) and self.device.type != 'cpu'  # 判断是否使用半精度
        self.model.half() if half else self.model.float()  # 设置模型的精度
        cudnn.benchmark = True  # 设置为True以加速固定图像大小的推理

    def detect(self, img, conf_thres=0.02, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=100):
        # 对输入图像进行预处理并进行推理
        im = self.letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # 将图像从HWC格式转换为CHW格式，同时从BGR转换为RGB
        im = np.ascontiguousarray(im)
        im = self.torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # 将图像数据类型转换为半精度或全精度
        im /= 255  # 归一化图像数据
        if len(im.shape) == 3: im = im[None]  # 为批次维度扩展图像
        
        pred = self.model(im)  # 进行推理
        det = self.non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 应用非极大值抑制
        det = det[0]
        detection = []
        if len(det):
            # 将检测结果转换为(x, y, w, h)格式并存储在detection列表中
            det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            for *box, conf, cls in reversed(det):
                label = self.names[int(cls)]
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w = xmax - xmin
                h = ymax - ymin
                x = xmin + (w // 2)
                y = ymin + (h // 2)
                detection.append((label, float(conf), (x, y, w, h)))
        return detection

    def close_cuda(self):
        # 清理CUDA缓存
        self.torch.cuda.empty_cache()
      



def ExtractDetection(detections, image, detectorParm, RoadData={}, calibrParm={}, e=None, sysParm={}):
    # img = image.copy()
    detetctedBox = list()
    detetcted = list()
    all_detected = []
    if len(detections) > 0:  
        id = 0
        for detection in detections:
            confident = detection[1]
            name_tag = str(detection[0])
            if name_tag in detectorParm['Classes']:

                if confident < detectorParm['Confidence'][name_tag]: continue

                x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3] 
                xmin, ymin, xmax, ymax = Utils.xywh2cord(float(x), float(y), float(w), float(h))
                x, y, w, h = Utils.cord2xywh(xmin, ymin, xmax, ymax)

                position = (x,ymax)
                onehot = [0,0,0,0,0,0,0]
                
                if sysParm.get('Use Road Mask for Ignore', False):
                    if not (position[1] >= RoadData['ROI Mask'].shape[0] or position[0] >= RoadData['ROI Mask'].shape[1]):
                        if RoadData['ROI Mask'][position[1], position[0], 0] == 0: continue
                
                if sysParm.get('Use BEV Mask for Ignore', False): 
                    position_bird = e.projection_on_bird(Calibration.applyROIxy(position, calibrParm['Region of Interest']))
                    try: 
                        if RoadData['Road Mask'][position_bird[1], position_bird[0]] == 0: continue
                    except: continue

                if name_tag == 'person':
                    if ymax - ymin > 200: continue
                    onehot = [1,0,0,0,0,0,0]
                if name_tag == 'car':onehot = [0,1,0,0,0,0,0]
                if name_tag == 'umbrella':onehot = [0,0,1,0,0,0,0]
                if name_tag == 'truck':onehot = [0,0,0,1,0,0,0]
                if name_tag == 'motorcycle':onehot = [0,0,0,0,1,0,0]
                if name_tag == 'bicycle':onehot = [0,0,0,0,0,1,0]
                if name_tag == 'bus': onehot = [0,0,0,0,0,0,1]

                id +=1
                detetctedBox.append([int(xmin), int(ymin), int(xmax), int(ymax), id, *onehot])
                all_detected.append([int(xmin), int(ymin), int(xmax), int(ymax), name_tag, id, confident])



        detetcted = np.array(detetctedBox) if len(detetctedBox) > 0 else np.empty((0, 12))

        
    return detetcted

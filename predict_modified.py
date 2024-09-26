# Combined Vehicle Speed, License Plate, and Speed Board Detection - predict.py

import hydra
import torch
import cv2
import math
import numpy as np
from collections import deque
from pathlib import Path
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Initialize global variables
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}
line = [(100, 500), (1050, 500)]
speed_line_queue = {}

# Initialize DeepSort Tracker
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,
                        n_init=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

# Estimate Vehicle Speed
def estimate_speed(Location1, Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 8  # pixels per meter
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6
    speed = d_meters * time_constant
    return int(speed)

# Convert bounding box coordinates to xywh
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

# Convert bounding box coordinates to tlwh
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = [int(i) for i in box]
        tlwh_bboxs.append([x1, y1, x2 - x1, y2 - y1])
    return tlwh_bboxs

# Get OCR from License Plate and Speed Board
def ocr_image(img, coordinates):
    x1, y1, x2, y2 = [int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])]
    cropped_img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    result = reader.readtext(gray)
    text = ""
    for res in result:
        if len(result) == 1 or (len(res[1]) > 2 and res[2] > 0.2):
            text = res[1]
    return text

# Draw bounding boxes and labels
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46, 162, 112), 3)
    height, width, _ = img.shape
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = f'{id}: {obj_name}'
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimate_speed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1
        try:
            label = f'{label} {sum(speed_line_queue[id]) // len(speed_line_queue[id])} km/h'
        except:
            pass

        text_ocr = ocr_image(img, (x1, y1, x2, y2))
        if text_ocr:
            label = f'{label} - Plate/Speed: {text_ocr}'

        draw_box(img, (x1, y1, x2, y2), label, color)

# Draw bounding box with label
def draw_box(img, coordinates, label, color):
    tl = 2
    x1, y1, x2, y2 = coordinates
    cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
    cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

# Determine the intersection between two lines
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Get direction of movement
def get_direction(point1, point2):
    direction_str = ""
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    return direction_str

# Assign a color to each object label
def compute_color_for_labels(label):
    color_map = {0: (85, 45, 255), 2: (222, 82, 175), 3: (0, 204, 255), 5: (0, 149, 255)}
    return color_map.get(label, [int((p * (label ** 2 - label + 1)) % 255) for p in palette])

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []

        for *xyxy, conf, cls in reversed(det):
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            if self.args.save_txt:
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                text_ocr = ocr_image(im0, xyxy)
                if text_ocr:
                    label = f'{label} - Plate/Speed: {text_ocr}'
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()

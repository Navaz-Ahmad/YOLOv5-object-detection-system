import os
import sys
import cv2
import torch
import numpy as np

# yolov5 path
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.insert(0, yolov5_path)

from utils.general import non_max_suppression
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

device = select_device('')
weights_path = os.path.join(os.path.dirname(__file__), 'yolov5', 'yolov5s.pt')
model = DetectMultiBackend(weights_path, device=device)
model.eval()


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32, scaleup=True):
    # Resize and pad image to new_shape, maintaining aspect ratio.
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width and height padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    # resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


def detect_objects_in_image(img_path, conf_thres=0.25, iou_thres=0.45):
    img0 = cv2.imread(img_path)
    assert img0 is not None, f"Image not found: {img_path}"

    img, ratio, (dw, dh) = letterbox(img0, new_shape=640, stride=32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # HWC to CHW, normalized
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    preds = model(img_tensor, augment=False)[0]

    # Apply NMS
    preds = non_max_suppression(preds, conf_thres, iou_thres)

    # Process detections
    if preds[0] is None:
        # No detections
        result_img = img0.copy()
    else:
        det = preds[0]
        # Rescale boxes from img_size to original image size
        det[:, :4] = scale_coords(img.shape[:2], det[:, :4], img0.shape[:2], ratio_pad=(ratio, (dw, dh)))

        result_img = img0.copy()
        for *xyxy, conf, cls in reversed(det):
            # Draw bounding box and label on original image
            label = f'{int(cls)} {conf:.2f}'
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(result_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(result_img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result image
    os.makedirs("static/results", exist_ok=True)
    result_path = os.path.join("static/results", f"detected_{os.path.basename(img_path)}")
    cv2.imwrite(result_path, result_img)

    return result_path

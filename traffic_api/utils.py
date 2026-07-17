import os
import logging
from datetime import datetime
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


def class_specific_nms(boxes, scores, class_ids, iou_thresh=0.6):
    """
    同类 NMS：仅在相同类别 ID 内抑制重叠框
    """
    if len(boxes) == 0:
        return []
    
    keep = []
    for cid in np.unique(class_ids):
        mask = class_ids == cid
        idx = np.where(mask)[0]
        cb = boxes[idx]
        cs = scores[idx]
        x1, y1, x2, y2 = cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cs.argsort()[::-1]
        
        while order.size > 0:
            i = order[0]
            keep.append(idx[i])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
    
    return keep

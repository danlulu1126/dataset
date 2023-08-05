import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torchvision.transforms as transforms


# 对标签坐标进行转换及执行扩充操作

# function: 将中心坐标（x,y），宽高（w, h）转为左上角坐标和右下角坐标的形式（x1, y1, x2, y2）
def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        img, boxes = data

        # convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape
        )

        # apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes
        )

        bounding_boxes = bounding_boxes.clip_out_of_image()

        # convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # w, h, _ = img.shape
        h = img.shape[0]
        w = img.shape[1]
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # w, h, _ = img.data
        h = img.shape[0]
        w = img.shape[1]

        # boxes[:, [1, 3]]*=h
        # print(boxes[:, [1, 3]])
        boxes[:, 1] = boxes[:, 1] * w
        boxes[:, 3] = boxes[:, 3] * w
        # boxes[:, [2, 4]]*=w
        boxes[:, 2] = boxes[:, 2] * h
        boxes[:, 4] = boxes[:, 4] * h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # extract image as pytorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels()
])

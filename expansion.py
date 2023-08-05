import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
from PIL import Image
from augmentations import *  # 这里的augmentations就是上面的augmentations.py文件


# 获取文件下属性为imgProperty的所有文件
def GetImgNameByEveryDir(file_dir, imgProperty):
    FileName = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in imgProperty:
                FileName.append(file)  # 保存图片名称
    return FileName


def readBoxes(txt_path):
    boxes = []
    with open(txt_path) as file:
        txt_lines = file.readlines()
        for txt_line in txt_lines:
            box = txt_line.rstrip().split(" ")
            boxes.append([int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])])

    return boxes


# 程序入口
# --img_path为需要扩增的图像数据
# note:图像和标签文件存在同一个文件夹
# 标签坐标：中心点坐标和宽高（cX, cY, W, H）

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default指定存放图像和标签的路径
    parser.add_argument("--img_path", type=str, default="D:\Desktop\lunwen\data_expansion\images+txt",
                        help='image path')
    opt = parser.parse_args()

    img_list = GetImgNameByEveryDir(opt.img_path, ['.jpg', '.jpeg','.png'])
    for img_name in tqdm(img_list):
        img_is_ok = 1
        boxes = []
        img_path = opt.img_path + '\\' + img_name
        try:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            img1 = cv2.imread(img_path)
        except Exception as e:
            print(f"could not read image '{img_path}'. ")
            img_is_ok = 0
        if img_is_ok:  # 如果图像存在，读取对应的标签文件
            txt_path = img_path[:-3] + 'txt'
            boxes = readBoxes(txt_path)
            print(boxes)
        transform = AUGMENTATION_TRANSFORMS
        boxes = np.array(boxes)
        temp_boxes = np.zeros_like(boxes)
        temp_boxes[:, :] = boxes[:, :]
        #
        # copy_num为对同一张图片扩充的张数
        copy_num = 4
        for i in np.arange(copy_num):
            new_img, bb_target = transform((img1, boxes))
            save_name = img_name[:-4] + "_" + str(i+1)
            cv2.imwrite(save_name + '.jpg', new_img)
            txt_file = open(save_name + '.txt', 'w')
            for line in bb_target:
                bb = str(int(line[0])) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str(
                    line[4]) + '\n'
                txt_file.write(bb)
            txt_file.close()
            boxes[:, :] = temp_boxes[:, :]


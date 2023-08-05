import imgaug.augmenters as iaa
from transforms import *  # 这里的transforms是上面的transforms.py文件，如果修改了文件名，这里对应修改即可


# imgaug是一个用于机器学习实验中图像增强的python库

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([  # 定义变换序列, 可根据需要自行增减或修改参数
            # iaa.Dropout([0.0, 0.01]),  # 随机去掉一些像素点，即把这些像素点变成0
            # iaa.Sharpen((0.0, 0.2)),  # 锐化处理
            iaa.Affine(rotate=(-60, 60), translate_percent=(0, 0)),
            # 仿射变换， rotate by -45 to 45 degrees (affects segmaps)
            # iaa.AddToBrightness((-30, 30)),  # 改变亮度
            # iaa.AddToHue((-20, 20)),  # 色调随机
            # # iaa.Sometimes(0.5, iaa.Fliplr(1)),  0.5概率翻转
            # iaa.Fliplr(1),  # 翻转图片，水平翻转图像（左右）
        ], random_order=True)


AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),  # 绝对标签
    DefaultAug(),  # 一些基本的数据增强
    RelativeLabels(),  # 相对标签
])

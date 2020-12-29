import albumentations as A
import random
import cv2
from albumentations import (
   RandomRotate90, IAAAdditiveGaussianNoise, GaussNoise, Compose, OneOf
)

vg = cv2.imread("assets/vg.jpeg")

transform = A.Compose([

    #剪裁 大小变换
    # A.CenterCrop(height=280, width=280, p=0.5), # 中心剪裁, 原图比这个小的话会报错
    # A.Resize(224,224),
    # A.RandomCrop(224,224),
    # A.RandomResizedCrop(224,224), # Torchvision's variant of crop a random part of the input and rescale it to some size.
    # A.RandomSizedBBoxSafeCrop(224,224),
    # A.RandomSizedCrop(min_max_height=(180,180),height=224,width=224), # Crop a random part of the input and rescale it to some size.
    # A.CropNonEmptyMaskIfExists(224,224), # Crop area with mask if mask is non-empty, else make random crop.


    # 图像质量变换
    A.Blur(blur_limit=(3,8), p=0.5), # 模糊
    A.Downscale(p=1.0, scale_min=0.7, scale_max=0.9), # 通过缩小再放大的方式降低图像质量。
    A.GaussNoise(),
    A.GaussianBlur(),
    A.GlassBlur(),
    A.MedianBlur(),
    A.MotionBlur(),

    # 颜色变换
    A.CLAHE(clip_limit=(1, 13), tile_grid_size=(8, 8), p=0.5), #对比度受限的自适应直方图均衡化
    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=random.randint(0,255), p=0.5), #随机丢弃输入通道
    A.ChannelShuffle(always_apply=False, p=0.5), # RGB通道随机重排
    A.ColorJitter(), # Randomly changes the brightness, contrast, and saturation of an image
    A.HueSaturationValue(), # Randomly change hue, saturation and value of the input image.
    A.ToGray(),

    # 风格变换
    A.HistogramMatching([vg],read_fn=lambda x: x), # 将原图变换到目标图到直方图
    A.FDA([vg],read_fn=lambda x: x), # 傅立叶域变换，这个比较牛，可以按照别的图像的风格进行变换

    # A.CoarseDropout(p=1.0, max_holes=100, max_height=10, max_width=10, min_holes=1, min_height=1, min_width=1),#挖掉一些长方形的孔
    A.Equalize(), # Equalize the image histogram.
    A.FancyPCA(),#Augment RGB image using FancyPCA from Krizhevsky's paper
    A.FromFloat(),
    A.IAAAdditiveGaussianNoise(),
    A.IAAEmboss(),
    A.IAASharpen(),
    A.IAASuperpixels(),
    # A.ISONoise(), # Apply camera sensor noise.
    A.ImageCompression(),
    A.InvertImg(),
    A.MultiplicativeNoise(),
    A.Normalize(),
    A.Posterize(),
    A.RGBShift(),
    A.RandomBrightnessContrast(),
    A.RandomFog(),
    A.RandomGamma(),
    A.RandomRain(),
    A.RandomShadow(),
    A.RandomSnow(),
    A.RandomSunFlare(),
    A.Solarize(),
    A.ToFloat(),
    A.ToSepia(),
    A.ElasticTransform(),
    A.Flip(),
    A.GridDistortion(),
    A.GridDropout(),
    A.HorizontalFlip(),
    A.IAAAffine(),
    A.IAACropAndPad(),
    A.IAAFliplr(),
    A.IAAFlipud(),
    A.IAAPerspective(),
    A.IAAPiecewiseAffine(),
    A.Lambda(),
    A.LongestMaxSize(),
    A.MaskDropout(),
    A.NoOp(),
    A.OpticalDistortion(),
    A.PadIfNeeded(),
    A.RandomCropNearBBox(),
    A.RandomGridShuffle(),
    A.RandomRotate90(),
    A.RandomScale(),
    A.Rotate(),
    A.ShiftScaleRotate(),
    A.SmallestMaxSize(),
    A.Transpose(),
    A.VerticalFlip(),
    OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)
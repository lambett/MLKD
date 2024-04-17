# copyright: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.
# 他应该是在你找的那篇FixMatch-pytorch复现的基础上做了少量修改，也算间接验证了可行性吧
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter
import random

# 全局变量
IMG_SIZE = 32

def RandomEqualize(img, val):
    # 以给定概率随机均衡给定图像的直方图，默认值0.5
    return transforms.RandomEqualize(p=0.8)(img)

def RandomAutocontrast(img, val):
    return transforms.RandomAutocontrast(val)(img)

def RandomAdjustSharpness(img, val):
    return transforms.RandomAutocontrast(val)(img)

def CenterCrop(img, val):
    # 从中心裁剪，以中心为中心，裁剪出来一个val*val的正方形
    val = int(val)
    return transforms.CenterCrop(val)(img)

def pad(img, val):
    # 就是padding，默认周边填0
    val = int(val)
    return transforms.Pad(val)(img)

def RandomCrop(img, val):
    # val表示概率
    w, h = img.size
    val = int(min(w, h) * val)
    return transforms.RandomCrop(val)(img)

def RandomHorizontalFlip(img, val):
    # val表示概率，以一定概率水平翻转
    return transforms.RandomHorizontalFlip()(img)

def RandomVerticalFlip(img, val):
    # val表示概率，以一定概率竖直翻转
    return transforms.RandomVerticalFlip(val)(img)

def RandomResizedCrop(img, val):
    # val表示概率，第一个参数输出图像大小，第二个参数裁剪面积上下限
    scale = (0.3, 0.9)
    return transforms.RandomResizedCrop(IMG_SIZE, scale=scale)(img)

def FiveCrop(img, val):
    # 上下左右中裁剪5个块出来
    val = int(val)
    return transforms.FiveCrop(val)(img)

def LinearTransformation(img, val):
    # 可以进行白化操作，感觉还是挺有意思的，但是一个transformation_matrix and mean_vector需要计算并作为参数
    val = int(val)
    return transforms.LinearTransformation()(img)

def ColorJitter(img, val):
    # 这四个参数可比较猛呐
    bri = (0.2, 4.0)    # 亮度调整范围
    con = (0.2, 4.0)    # 调节对比度的参数
    sat = (0.2, 4.0)    # 调饱和度的参数
    hue = (-0.5, 0.5)   # 调颜色或者色调的参数
    return transforms.ColorJitter(brightness=bri, contrast=con, saturation=sat, hue=hue)(img)

def RandomRotation(img, val):
    # 可以进行白化操作，感觉还是挺有意思的，但是一个transformation_matrix and mean_vector需要计算并作为参数
    val = int(val)
    return transforms.RandomRotation(val)(img)

def RandomErasing(img, val):
    # 可以进行白化操作，感觉还是挺有意思的，但是一个transformation_matrix and mean_vector需要计算并作为参数
    val = int(val)
    img = transforms.PILToTensor()(img)
    img = transforms.RandomErasing(p=0.8)(img)
    return transforms.ToPILImage()(img)

def RandomAffine(img, val):
    # 仿射变换和裁剪和旋转还是非常不一样的
    val = int(val)
    # img = transforms.PILToTensor()(img)
    deg = 30
    she = [-45, 45, -45, 45]
    return transforms.RandomAffine(deg, shear=she)(img)

def RandomPerspective(img, val):
    # 这个透视变换不知道啥意思，跟电影院看电影似的
    # val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.RandomPerspective(distortion_scale=val, p=0.8)(img)

def GaussianBlur(img, val):
    # 这个透视变换不知道啥意思，跟电影院看电影似的
    val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.GaussianBlur(kernel_size=11)(img)

def RandomGrayscale(img, val):
    # 以概率p转换为灰度图
    val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.RandomGrayscale(p=0.8)(img)

def RandomInvert(img, val):
    # 以概率p反转图像
    val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.RandomInvert(p=0.7)(img)

def RandomPosterize(img, val):
    # 以概率p反转图像
    val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.RandomPosterize(bits=val, p=0.8)(img)

def RandomSolarize(img, val):
    # 以概率p反转图像, 所有等于或高于此值的像素都被反转。
    val = int(val)
    # img = transforms.PILToTensor()(img)
    return transforms.RandomSolarize(threshold=val, p=0.8)(img)

def low_freq_mutate_np(amp_src, amp_tar, L=0.1):

    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_tar = np.fft.fftshift(amp_tar, axes=(-2, -1))
    # np.fft.fftshift(img)  将图像中的低频部分移动到图像的中心
    # 参数说明：img表示输入的图片
    _src, h_src, w_src = a_src.shape
    # _tar, h_tar, w_tar = a_tar.shape
    b = (np.floor(np.amin((h_src, w_src))*L)).astype(int)
    b = max(b, 1)
    # np.amin()取数组最小值
    h_central_src = np.floor(h_src/2.0).astype(int)
    w_central_src = np.floor(w_src/2.0).astype(int)
    # 源域的替换起止坐标
    h1 = h_central_src-b
    h2 = h_central_src+b+1
    w1 = w_central_src-b
    w2 = w_central_src+b+1
    # 目标域中心坐标
    # h_central_tar = np.floor(h_tar / 2.0).astype(int)
    # w_central_tar = np.floor(w_tar / 2.0).astype(int)
    # 目标域的替换起止坐标，当图片尺寸不一致时使用
    # h1_tar = h_central_tar - b
    # h2_tar = h_central_tar + b + 1
    # w1_tar = w_central_tar - b
    # w2_tar = w_central_tar + b + 1
    a_src[:, h1:h2, w1:w2] = a_tar[:, h1:h2, w1:w2]
    # a_src[:, h1:h2, w1:w2] = 0
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img=None, trg_img=None, L=0.00005):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))
    # axes=(-2, -1)的意思是在最后两个维度上计算二维离散傅立叶变换
    # 返回的值fft_src_np为复数

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_tar, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
    # np.abs()计算复数的模，np.angle()计算复数的角度

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src=amp_src, amp_tar=amp_tar, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    # np.real()取实部
    return src_in_trg


def prosess_image(img_src=None, img_tar=None):
    img_src = np.array(img_src)    # 维度信息为（667，1000，3）
    img_src = img_src.transpose((2, 0, 1))
    img_tar = np.array(img_tar)
    img_tar = img_tar.transpose((2, 0, 1))
    src_in_trg = FDA_source_to_target_np(src_img=img_src, trg_img=img_tar, L=0.01)
    src_in_trg = src_in_trg.transpose((1, 2, 0))
    # src_in_trg = (src_in_trg - np.min(src_in_trg))/np.ptp(src_in_trg)*255
    # 放缩到0-255之间
    src_in_trg[src_in_trg < 0] = 0
    src_in_trg[src_in_trg > 255] = 255
    # Image.fromarray(src_in_trg.astype('uint8')).convert('RGB').save('demo_images/test000.jpg')
    Image.fromarray(src_in_trg.astype('uint8')).convert('RGB').show()

def transforms_list():
    # 参数说明，[-1,0]表示参数误无用，写在函数实现里面
    l = [
        # 挺好的操作，但是暂时实现不了
        (LinearTransformation, 0, 1),
        # 常规操作
        (FiveCrop, 0.1, 0.25),
        (RandomHorizontalFlip, -1, 0),
        (RandomCrop, 0.7, 0.9),
        (pad, 10, 20),
        (CenterCrop, 10, 20),
        (RandomAdjustSharpness, 0.3, 0.7),
        (RandomAutocontrast, 0.3, 0.7),
        (RandomCrop, 0.7, 0.9),
        # 保留操作
        (ColorJitter, -1, 0),    # 调整对比度、亮度、色调等
        (RandomResizedCrop, 0.3, 0.9),  # 根据面积计算概率
        (RandomRotation, 0, 90),    # 旋转
        (RandomErasing, -1, 0),  # 随机抹除
        (RandomAffine, -1, 0),  # 仿射变换
        (RandomPerspective, 0.3, 0.7),  # 透视变换，跟电影院看电影似的
        (GaussianBlur, -1, 0),  # 高斯模糊
        (RandomGrayscale, -1, 0),   # 随机转换灰度图
        (RandomInvert, -1, 0),  # 有点像医院拍片子
        (RandomPosterize, 2, 5),    # 就是用几个bit的数据表示每个通道像素值
        (RandomSolarize, 196, 256), # 高于一定值的像素就会反转
        (RandomEqualize, -1, 0), # 均衡直方图
    ]
    return l

def before_resize_list():
    # 参数说明，[-1,0]表示参数误无用，写在函数实现里面
    l = [
        (ColorJitter, -1, 0),    # 调整对比度、亮度、色调等
        (RandomPosterize, 2, 5),  # 就是用几个bit的数据表示每个通道像素值
        (RandomSolarize, 196, 256),  # 高于一定值的像素就会反转
        (RandomGrayscale, -1, 0),   # 随机转换灰度图
        (RandomInvert, -1, 0),  # 有点像医院拍片子
        (RandomEqualize, -1, 0), # 均衡直方图
    ]
    return l

def after_resize_list():
    # 参数说明，[-1,0]表示参数误无用，写在函数实现里面
    l = [
        (RandomErasing, -1, 0),  # 随机抹除
        (GaussianBlur, -1, 0),  # 高斯模糊
        (RandomRotation, 0, 90),  # 旋转
        (RandomAffine, -1, 0),  # 仿射变换
        # (RandomPerspective, 0.3, 0.7),  # 透视变换，跟电影院看电影似的；透视变换老报错，暂时废掉。。
        (RandomHorizontalFlip, -1, 0),
        # (pad, 1, 25), # padding有问题，他是在外面加一圈
    ]
    return l
#####################################################################################
# transform的过程：进行跟尺寸无关的操作-->crop并且resize至224-->进行跟尺寸相关的操作-->傅里叶变换

class RandAugment_before(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = before_resize_list()

    def __call__(self, img):
        random.shuffle(self.augment_list)
        ops = random.sample(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            # random.random()随机生成0-1之间的浮点数
            img = op(img, val)
        return img


class RandAugment_after(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = after_resize_list()

    def __call__(self, img):
        random.shuffle(self.augment_list)
        ops = random.sample(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            # random.random()随机生成0-1之间的浮点数
            img = op(img, val)
        return img

class RandAugment_fft(object):
    def __init__(self, L=0.01):
        self.L = L

    def __call__(self, img):
        random.shuffle(self.augment_list)
        ops = random.sample(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            # random.random()随机生成0-1之间的浮点数
            img = op(img, val)
        return img

class RandAugment_test:
    def __init__(self, order):
        self.o = order
        self.augment_list = transforms_list()

    def __call__(self, img):
        ops = self.augment_list[self.o]
        ops = list(ops)
        op, min_val, max_val = ops[0], ops[1], ops[2]
        val = min_val + float(max_val - min_val) * random.random()
        # random.random()随机生成0-1之间的浮点数
        img = op(img, val)
        return img


class ResizeImage(object):
    """Resize the input PIL Image to the given size.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class GaussianBlur(object):
    def __init__(self, sigma=[.1, .2]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

if __name__ == '__main__':
    randaug = RandAugment_test(0)
    # print(randaug)
    # for item in randaug.augment_list:
    #     print(item)
    im = Image.open("1.jpg")
    # src.jpg: (1000,667)
    im = transforms.Resize(224)(im)
    im = randaug(im)
    im.show()
    print('test over ~~~')
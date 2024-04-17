import os
from .resnet import (
    resnet8,
    resnet14,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet8x4,
    resnet32x4,
)
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn

from .resnet56_M_r20 import res56_M_r20
from .resnet56_M_r32 import res56_M_r32
from .resnet56_M_r44 import res56_M_r44

from .vgg16_M_r20 import vgg16_M_r20
from .vgg16_M_r44 import vgg16_M_r44
from .vgg16_M_r32 import vgg16_M_r32


cifar10_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/cifar10_teachers/"
)
cifar10_model_dict = {
    # teachers
    "resnet56": (
        resnet56,
        cifar10_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "vgg16": (vgg16_bn, cifar10_model_prefix + "vgg16_vanilla/ckpt_epoch_240.pth"),
    # students
    "res56_M_r20": (res56_M_r20, None),
    "res56_M_r32": (res56_M_r32, None),
    "res56_M_r44": (res56_M_r44, None),
    "vgg16_M_r20": (vgg16_M_r20, None),
    "vgg16_M_r32": (vgg16_M_r32, None),
    "vgg16_M_r44": (vgg16_M_r44, None),
}

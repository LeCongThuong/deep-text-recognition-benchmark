import utils.ocrodeg as ocrodeg
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import scipy.ndimage as ndi
import math


def rotate_word(image, min_angle=-10, max_angle=10):
    angle = random.randint(min_angle, max_angle)
    transformed_image = ocrodeg.transform_image(image, angle=angle * math.pi / 180)
    return transformed_image


class RotateWord(ImageOnlyTransform):
    def __init__(self, min_angle, max_angle, always_apply=False, p=1):
        super(RotateWord, self).__init__(always_apply, p)
        self.min_angle = min_angle
        self.max_angle = max_angle

    def apply(self, img, **params):
        return rotate_word(img, self.min_angle, self.max_angle)


def space_between_chacracter(image, aniso_list):
    aniso = aniso_list[random.randrange(len(aniso_list))]
    transformed_image = ocrodeg.transform_image(image, aniso=aniso)
    return transformed_image


class SpaceCharacter(ImageOnlyTransform):
    def __init__(self, aniso_list, always_apply=False, p=1):
        super(SpaceCharacter, self).__init__(always_apply, p)
        self.aniso_list = aniso_list

    def apply(self, img, **params):
        return space_between_chacracter(img, self.aniso_list)


def scale_word(image, scale_list):
    scale = scale_list[random.randrange(len(scale_list))]
    return ocrodeg.transform_image(image, scale=scale)


class ScaleWord(ImageOnlyTransform):
    def __init__(self, scale_list, always_apply=False, p=1):
        super(ScaleWord, self).__init__(always_apply, p)
        self.scale_list = scale_list

    def apply(self, img, **params):
        return scale_word(img, self.scale_list)


def distort_with_noise(image, sigma_list):
    sigma = sigma_list[random.randrange(len(sigma_list))]
    noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, 5.0)
    distorted = ocrodeg.distort_with_noise(image, noise)
    return distorted


class DistortWithNoise(ImageOnlyTransform):
    def __init__(self, sigma_list, always_apply=False, p=1):
        super(DistortWithNoise, self).__init__(always_apply, p)
        self.scale_list = sigma_list

    def apply(self, img, **params):
        return distort_with_noise(img, self.scale_list)


def gaussian_filter(image, s_list):
    s = s_list[random.randrange(len(s_list))]
    blurred = ndi.gaussian_filter(image, s)
    return blurred


class GaussianFilter(ImageOnlyTransform):
    def __init__(self, s_list, always_apply=False, p=1):
        super(GaussianFilter, self).__init__(always_apply, p)
        self.s_list = s_list

    def apply(self, img, **params):
        return gaussian_filter(img, self.s_list)
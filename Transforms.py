from monai.transforms import MapTransform
import torch
from HelperFunctions import *
import numpy as np
import random


# custom to tensor transform
class ToTensor(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.Tensor(data[key])

        return data


# permute the datasets into tensor compatible format
class PermutateTransform(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].permute(0, 3, 1, 2)

        return data


# Used for debugging
class TestCompose(object):
    def __init__(self, message=''):
        self.message = message

    def __call__(self, data):
        img, label = data[DataType.Image], data[DataType.Label]

        print(f'{self.message} img.shape {img.shape} label.shape {label.shape}')
        print(type(data))
        return data

    # Convert labels into three channel : background, pancreas and tumour


# The pancreas and tumour are classified together as one
class ConvertToMultiChannelBasedOnLabelsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the pancreas
    label 2 is the cancer
    The possible classes are Pancreas, Cancerous

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []

            d[key] = np.squeeze(d[key])

            # label 0 is Background
            result.append(d[key] == 0)

            # merge label 1 and 2 to construct Pancreas and Cancer
            # result.append(np.logical_or(d[key] == 1, d[key] == 2))
            result.append(d[key] == 1)

            # label 2 is Cancer
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


# Randomw Window Intensity to augment the window intensity for the input images
# This is no longer used in favour of ManualWindowIntensity class
class RandomWindowIntensity(MapTransform):
    def __init__(self, keys, thresholds, prob=1):
        super(RandomWindowIntensity, self).__init__(keys)

        self.thresholds = thresholds
        self.prob = prob

    def __call__(self, data):
        d = dict(data)

        to_transform = random.uniform(0, 1) < self.prob

        if not to_transform:
            return d

        rand_threshold_idx = random.randint(0, len(self.thresholds) - 1)
        print(f'shape at ranfom window is {d[self.keys[0]].shape}')
        d[self.keys[0]] = np.clip(d[self.keys[0]], -self.thresholds[rand_threshold_idx],
                                  self.thresholds[rand_threshold_idx])
        return d


#
class ManualWindowIntensity(MapTransform):
    '''
        return:
    '''

    def __init__(self, keys):
        super(ManualWindowIntensity, self).__init__(keys)

        # Hardcoded thresholds
        self.thresholds = [(-150, 350), (-20, 125), (60, 170), (-40, 150)]

    def __call__(self, data):
        d = dict(data)

        img = d[self.keys[0]]

        # Window images according to specified thresholds
        img2 = np.clip(img, self.thresholds[0][0], self.thresholds[0][1])
        img3 = np.clip(img, self.thresholds[1][0], self.thresholds[1][1])
        img4 = np.clip(img, self.thresholds[2][0], self.thresholds[2][1])
        img5 = np.clip(img, self.thresholds[3][0], self.thresholds[3][1])

        # Combine as a multichannel images - similar to feeding in multimodality input
        multi_channel_img = np.concatenate((img, img2, img3, img4, img5), axis=0)
        d[self.keys[0]] = multi_channel_img
        return d


# Transform to add patient ID in dictionary - allows easy patient searching
class AddSubjectId(object):
    def __call__(self, data):
        data[DataType.Id] = extract_subj_id(data[DataType.Image], is_test=True)

        return data

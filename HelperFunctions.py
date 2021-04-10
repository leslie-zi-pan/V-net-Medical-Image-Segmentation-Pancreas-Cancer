import json
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import torch
import nibabel as nib

# Match Data type to json data information file
class DataType():
    Image = 'image'
    Label = 'label'
    Id = 'subj_id'
    ModelType = 'model_type'


# Enum to get convolution sample type
class ResampleTypeEnum(Enum):
    Upsample = 1
    Downsample = 2


# extracts subject id from path name
def extract_subj_id(string_path, is_test=False):
    datatype = DataType.Image if DataType.Image in string_path else DataType.Label
    test_train_str = 'Ts' if is_test else 'Tr'

    result = string_path.replace(f'./{datatype}s{test_train_str}/pancreas_', '').replace('.nii.gz', '')
    return result


# Prints the shapes of a list of images
def print_shape(*images):
    for idx, image in enumerate(images):
        print(f'Item {idx} Shape \t{image.shape}')


# View a slice of an image
def view_slice(slice, title='', gray=False):
    plt.title(title)

    cmap = None

    if gray:
        cmap = 'gray'

    plt.imshow(slice, cmap=cmap)
    plt.show()


# Get a dictionary from a list_dict filtered by a spcific key value
def get_dicts_from_dicts(dicts, key, value):
    x = [item for item in dicts if any(map(item[key].__contains__, value))]

    return x


# Get a dictionary from a list_dict filtered by a spcific key value
def get_dict_from_dicts(dicts, key, value):
    x = next(item for item in dicts if item[key] == value)
    return x


# Convert torch tensor to numpy array
def numpy_from_tensor(x):
    return x.detach().cpu().numpy()


# Used for debugging and better memory management
def print_available_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    return f


# Convert Haobao's nifty to suitable dimension
def convert_haobo_nib(filepath):
    data = nib.load(filepath)
    data = data.get_fdata()
    data = np.array(torch.tensor(data).permute(3, 0, 4, 1, 2)).squeeze()

    return nib.Nifti1Image(data, np.eye(4))


# Convert tensor image to nibabel format
def tensor_to_nib(tensor_data):
    return nib.Nifti1Image(np.array(tensor_data), np.eye(4))

def get_json_file(path, filename):
    with open(f'{path}/{filename}') as json_file:
        file = json.load(json_file)

    return file
from HelperFunctions import *
from Transforms import *
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    ToTensord,
)
from DataSet import *
from VNet import *
from Training import *

train_transform = Compose(
    [
        LoadImaged(keys=[DataType.Image, DataType.Label]),
        AddChanneld(keys=[DataType.Image, DataType.Label]),
        CropForegroundd(keys=[DataType.Image, DataType.Label], source_key=DataType.Image),
        RandCropByPosNegLabeld(
            keys=[DataType.Image, DataType.Label],
            label_key=DataType.Label,
            spatial_size=(110, 110, 25), # Crop to slightly larger than desired for elastic deformation - minimises errors created from padding
            pos=1,
            neg=0,
            num_samples=1,
            image_key=DataType.Image,
            image_threshold=0,
        ),
        # This function handles both affine and elastic deformation together
        # To minimise padding issues, we will elastic and affine transform first on original before any cropping
        Rand3DElasticd(
            keys=[DataType.Image, DataType.Label],
            sigma_range=(5, 8), # Sigma range for elastic deformation
            magnitude_range=(100, 200), # maginitude range for elastic deformation
            mode=('bilinear', 'nearest'),
            # Output to desired crop size
            spatial_size=(96, 96, 16),
            # Probability of augmentation - we will kepp most unchanged and augment 30%
            prob=0.4,
            # Only rotate depthwise as patient generally only varies in the depthwise direction
            rotate_range=(0, 0, np.pi/15),
            # Set translation to 0.1
            translate_range =(0.1, 0.1, 0.1),
            # scale in all direction by 0.1
            scale_range=(0.1, 0.1, 0.1)
        ),
        ManualWindowIntensity(keys=DataType.Image),
        # RandomWindowIntensity(keys=DataType.Image, thresholds=[1024, 512, 256, 128], prob=0.8), # Commented out, manual windowing used instead
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=DataType.Label),
        ToTensord(keys=[DataType.Image, DataType.Label]),
        PermutateTransform(keys=[DataType.Image, DataType.Label]),
    ])

val_transform = Compose(
    [
        LoadImaged(keys=[DataType.Image, DataType.Label]),
        AddChanneld(keys=[DataType.Image, DataType.Label]),
        CropForegroundd(keys=[DataType.Image, DataType.Label], source_key=DataType.Image),
        RandCropByPosNegLabeld(
            keys=[DataType.Image, DataType.Label],
            label_key=DataType.Label,
            spatial_size=(96, 96, 16),
            pos=1,
            neg=0,
            num_samples=1,
            image_key=DataType.Image,
            image_threshold=0,
        ),
        ManualWindowIntensity(keys=DataType.Image),
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=DataType.Label),
        ToTensord(keys=[DataType.Image, DataType.Label]),
        PermutateTransform(keys=[DataType.Image, DataType.Label]),
    ])

# Test transform same as validation transform but without the cropping - we want to feed in whole image size
test_transform =  Compose(
    [
        AddSubjectId(),
        LoadImaged(keys=[DataType.Image, DataType.Label]),
        AddChanneld(keys=[DataType.Image, DataType.Label]),
        ManualWindowIntensity(keys=DataType.Image),
        ConvertToMultiChannelBasedOnLabelsClassesd(keys=DataType.Label),
        ToTensord(keys=[DataType.Image, DataType.Label]),
        PermutateTransform(keys=[DataType.Image, DataType.Label]),
    ])


EPOCHS = 801
# Give background less weighting and more to tumour and pancreas
weights = [0.3, 1.0, 1.5]
CE_WEIGHTS = torch.FloatTensor(weights).cuda()

# Convert softmaxes into ont hot
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
post_label = AsDiscrete(to_onehot=True, n_classes=3)

if __name__ == '__main__':
    data_path = '/Data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    json_data = get_json_file(data_path, 'dataset.json')

    training_paths = get_dicts_from_dicts(json_data['training'], DataType.Image, json_data)
    testing_paths = [{DataType.Image: item} for item in json_data['test']]

    # split in 8:2 ratio
    train_cutoff = int(0.8 * len(training_paths))
    val_len = len(training_paths) - train_cutoff


    # create dataset from specified json paths
    train_dataset  = CustomCacheDataSet(training_paths, train_transform, 32)
    val_dataset = CustomCacheDataSet(testing_paths, val_transform, 16)
    # val_dataset2 = CustomCacheDataSet(training_paths[train_cutoff:], val_transform2, 1) # Used for tuning
    test_dataset = CustomCacheDataSet(testing_paths, test_transform, 1)

    # Setting up Data Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    train_loader_tuning = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Check losses against weight decay variation
    weights = np.logspace(-6, -2, 5)
    plots2 = tune_weight_decay_network(
        training_loader=train_loader_tuning,
        network=Vnet(),
        weights=weights
    )

    # Train the network
    net = train_network(
        training_loader=train_loader,
        val_loader=val_loader,
        network=Vnet(),
        pre_load_training=False,
        checkpoint_name='/model_ch_with_cancer_aug_with_window_5channlintensity_final1.pt'
    )

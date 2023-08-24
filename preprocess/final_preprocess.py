import numpy as np
from skimage import exposure
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ToTensord,
)
from monai.data import CacheDataset, Dataset, DataLoader
from monai.utils import set_determinism
from glob import glob
import os

def preprocess_image(image, a_min=-200, a_max=200):
    # Apply windowing to the image
    adjusted_image = (image - 60 + 200) / 400  # Adjust intensity values

    # Normalize the adjusted image to the range [0, 1]
    normalized_image = (adjusted_image - np.min(adjusted_image)) / (np.max(adjusted_image) - np.min(adjusted_image))

    # Apply CLAHE enhancement
    enhanced_image = exposure.equalize_adapthist(normalized_image)

    return enhanced_image

def prepare(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=False):

    set_determinism(seed=0)

    def combined_transforms(image):
        enhanced_image = preprocess_image(image, a_min=a_min, a_max=a_max)
        return enhanced_image

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]

    combined_train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            # Applying the combined preprocessing function here
            combined_transforms(keys=["vol"]),
            ToTensord(keys=["vol", "seg"]),
        ]
    )

    combined_test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            # Applying the combined preprocessing function here
            combined_transforms(keys=["vol"]),
            ToTensord(keys=["vol", "seg"]),
        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=combined_train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=combined_test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=combined_train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=combined_test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

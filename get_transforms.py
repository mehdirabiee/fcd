import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd,
    RandFlipd, RandRotated, RandAffined,
    RandBiasFieldd, RandAdjustContrastd, RandGaussianNoised,
    RandCoarseDropoutd, RandCropByPosNegLabeld, Invertd,
    Activationsd, AsDiscreted, RandShiftIntensityd, ResampleToMatchd, Spacingd
)
from utils import GridMaskd


from monai.transforms import MapTransform
import numpy as np

class ReplaceNaNd(MapTransform):
    """
    Replace NaN values in images or labels with a constant (default=0.0).
    """
    def __init__(self, keys, replacement=0.0):
        super().__init__(keys)
        self.replacement = replacement

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                arr = d[key]
                # works for both numpy arrays and torch tensors
                if isinstance(arr, np.ndarray):
                    arr = np.nan_to_num(arr, nan=self.replacement)
                else:  # torch tensor
                    arr = arr.nan_to_num(self.replacement)
                d[key] = arr
        return d


class FCDTrainTransform:
    def __init__(self, params):
        self.params = params
        self.normalize_intensity = False
        self.coarse_dropout_max_prob = self.params.get('coarse_dropout_max_prob', 0.0)
        self.coarse_dropout_start_epoch = self.params.get('coarse_dropout_start_epoch', 0.0)
        self.coarse_dropout = RandCoarseDropoutd(keys=["image"], holes=5, spatial_size=(16, 16, 16), fill_value=0, prob=0)
        
        self.gridmask_max_prob = self.params.get('gridmask_max_prob', 0.0)
        self.gridmask_start_epoch = self.params.get('gridmask_start_epoch', 0.0)
        self.gridmask = GridMaskd(keys=["image"], apply_prob=self.gridmask_max_prob, grid_spacing_range=(16, 32), mask_ratio=0.5, rotate_range=1, invert_mask=False)

        self.train_transforms = Compose([
            # Load and preprocess
            LoadImaged(keys=["image", "label"], image_only=False),
            ReplaceNaNd(keys=["image"], replacement=0.0),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Ensure label matches image spacing
            ResampleToMatchd(keys="label", key_dst="image", mode="nearest"),            
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, channel_wise=True
            ),
            
            # Crop based on lesion presence
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=params['patch_size'], pos=1, neg=1,
                num_samples=params['samples_per_case'], image_key=None, image_threshold=0
            ),

            # Spatial augmentations
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            #RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, mode=["bilinear", "nearest"], prob=0.5),
            RandRotated(keys=["image", "label"], range_y=torch.pi/2.0, mode=["bilinear","nearest"], prob=0.5),
            
            # Affine transforms (mild)
            #RandAffined(keys=["image", "label"], rotate_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1), 
            #            mode=["bilinear", "nearest"], padding_mode="zeros", prob=0.1),

            # Intensity augmentations
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.1, prob=0.5),
            #RandBiasFieldd(keys=["image"], coeff_range=(0.1, 0.3), degree=3, prob=0.1),
            #RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.1),

            # Coarse dropout for robustness
            self.coarse_dropout,
            self.gridmask,
        ])
        self.valid_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            ReplaceNaNd(keys=["image"], replacement=0.0),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Ensure label matches image spacing
            ResampleToMatchd(keys="label", key_dst="image", mode="nearest"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, channel_wise=True
            ),
        ])        
        if self.normalize_intensity:
            self.train_transforms.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))
            self.valid_transforms.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))
    
    def get_transforms(self):
        return self.train_transforms, self.valid_transforms

    def has_gradual_prob(self):
        return self.coarse_dropout_max_prob > 0 or self.gridmask_max_prob > 0
    
    def set_prob(self, epoch, max_epoch):
        if self.coarse_dropout_max_prob == 0 or epoch < self.coarse_dropout_start_epoch:
            self.coarse_dropout.prob = 0
        else:
            self.coarse_dropout.prob = self.coarse_dropout_max_prob * min(1, (epoch - self.coarse_dropout_start_epoch) / (max_epoch - self.coarse_dropout_start_epoch))
        
        if self.gridmask_max_prob == 0 or epoch < self.gridmask_start_epoch:
            self.gridmask.set_prob(0, 1)
        else:
            self.gridmask.set_prob(epoch - self.gridmask_start_epoch, max_epoch - self.gridmask_start_epoch)


def get_test_transforms(params):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=0.5, 
                upper=99.5, 
                b_min=0, 
                b_max=1, 
                clip=True, 
                channel_wise=True
            ),
        ]
    )

    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=False, softmax=True),
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
        ]
    )

    return test_transforms, post_transforms

def test(data_dir):
    from monai.data import Dataset, DataLoader
    from monai.utils import first

    from config import get_default_params
    from get_data import get_data

    params = get_default_params()
    transforms = FCDTrainTransform(params)
    train_transform, val_transform = transforms.get_transforms()

    data_dict = get_data(data_dir, params)
    dataset = Dataset(data=data_dict, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_data in dataloader:
        # Get the first transformed sample
        len = batch_data["image"].shape[0]
        print(len)

    transformed_sample = first(dataloader)
    image = transformed_sample["image"][0]
    label = transformed_sample["label"][0]

    # Convert tensors to numpy
    image_np = image.squeeze().cpu().numpy()
    label_np = label.squeeze().cpu().numpy()
    return image_np, label_np

if __name__ == "__main__":
    data_dir = "/mnt/d/dimes/dataset/ds004199_fsl/val/"
    test(data_dir)
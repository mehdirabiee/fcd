from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    RandRotate90d,
    RandGaussianNoised,
    RandRotated,
    SpatialPadd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    ScaleIntensityd,
    ToTensord,
    EnsureChannelFirstd,
    SpatialCropd,
    CenterSpatialCropd,
    Activationsd,
    AsDiscreted,
    Invertd,
    SelectItemsd,
    ConcatItemsd,
    RandCoarseDropoutd,
    RandAffined,
    RandBiasFieldd,
    RandGaussianSmoothd,
    RandSpatialCropd,
)

def get_test_transforms(params):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
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

def get_trainval_transforms(params):
    train_transforms = Compose(
        [
            # Basic loading and preprocessing
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=0.5, 
                upper=99.5, 
                b_min=0, 
                b_max=1, 
                clip=True, 
                channel_wise=True
            ),
            
            # Enhanced cropping with more control
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=params['patch_size'],
                pos=1,
                neg=1,
                num_samples=params['samples_per_case'],
                image_key=None,
                image_threshold=0,
            ),
            
            # Spatial transforms
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            RandRotated(
                keys=["image", "label"], 
                range_y=1.0/2*3.14159, 
                mode=["bilinear", "nearest"],
                prob=0.5
            ),
            
            # Additional spatial transforms
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=params['patch_size'],
                random_size=False,
                random_center=True,
            ),
            
            # Elastic deformation
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.05, 0.05, 0.05),
                scale_range=(0.1, 0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="zeros"
            ),
            
            # Intensity transforms
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.1, prob=0.5),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.2,
                prob=0.5
            ),
            
            # Additional intensity normalization
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=0.5,
                divisor=0.5,
                nonzero=True
            ),
            
            # MRI-specific transforms
            RandBiasFieldd(
                keys=["image"],
                prob=0.3,
                degree=3,
                coeff_range=(0.0, 0.1)
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                prob=0.3
            ),
            
            # Masking transforms
            RandCoarseDropoutd(
                keys=["image"],
                holes=5,
                spatial_size=[16, 16, 16],
                fill_value=0,
                prob=0.3
            ),
        ]
    )

    valid_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
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

    return train_transforms, valid_transforms
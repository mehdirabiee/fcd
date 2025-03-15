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
)

def get_test_transforms(params):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # Scale intensity for T1 and FLAIR channels
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=0.5, 
                upper=99.5, 
                b_min=0, 
                b_max=1, 
                clip=True, 
                channel_wise=True
            ),
            # If there's a thickness map (3rd channel), normalize it separately
            SelectItemsd(keys=["image"], channel_indices=[-1] if len(params.get('chans_in', [])) > 2 else None),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=6,  # typical max thickness in mm
                b_min=0,
                b_max=1,
                clip=True
            ) if len(params.get('chans_in', [])) > 2 else None,
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
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Scale intensity for T1 and FLAIR channels
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=0.5, 
                upper=99.5, 
                b_min=0, 
                b_max=1, 
                clip=True, 
                channel_wise=True
            ),
            # If there's a thickness map (3rd channel), normalize it separately
            SelectItemsd(keys=["image"], channel_indices=[-1] if len(params.get('chans_in', [])) > 2 else None),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=6,  # typical max thickness in mm
                b_min=0,
                b_max=1,
                clip=True
            ) if len(params.get('chans_in', [])) > 2 else None,
            
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
            
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            RandRotated(
                keys=["image", "label"], 
                range_y=1.0/2*3.14159, 
                mode=["bilinear", "nearest"],
                prob=0.5
            ),
            # Only apply intensity augmentations to T1 and FLAIR channels
            SelectItemsd(keys=["image"], channel_indices=[0, 1] if len(params.get('chans_in', [])) > 2 else None),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.1, prob=0.5),
        ]
    )

    valid_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Scale intensity for T1 and FLAIR channels
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=0.5, 
                upper=99.5, 
                b_min=0, 
                b_max=1, 
                clip=True, 
                channel_wise=True
            ),
            # If there's a thickness map (3rd channel), normalize it separately
            SelectItemsd(keys=["image"], channel_indices=[-1] if len(params.get('chans_in', [])) > 2 else None),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=6,  # typical max thickness in mm
                b_min=0,
                b_max=1,
                clip=True
            ) if len(params.get('chans_in', [])) > 2 else None,
        ]
    )

    return train_transforms, valid_transforms
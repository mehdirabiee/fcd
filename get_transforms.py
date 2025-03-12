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
)

def get_test_transforms(params):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            #CenterSpatialCropd(keys=["image"], roi_size=(160, 218, 182)), #fix w when sampleing 
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, channel_wise=True),
            #NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=False),
        ]
    )

    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=False, softmax=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=test_transforms,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
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
            #CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 218, 182)), #fix w when sampleing 
            ScaleIntensityRangePercentilesd(["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, channel_wise=True),
            #NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=False),
            
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
            RandRotated(keys=["image", "label"], range_y=1.0/2*3.14159, mode=["bilinear","nearest"], prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"],  std=0.1,     prob=0.5),    
        ]
    )

    valid_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            #CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 218, 182)), #fix w when sampleing 
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, channel_wise=True),
            #NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=False),
            
        ]
    )

    return train_transforms, valid_transforms
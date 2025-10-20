
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.transforms import LoadImage

def inspect_nifti_file_monai(data_file):
    image, meta = LoadImage(image_only=False)(data_file)

    #matshow3d(volume=image, title='3D Image Visualization', cmap='gray')
    #plt.show()

    # Get the affine transformation matrix
    affine = meta['affine'].numpy()
    print("Affine Matrix:\n", affine)

    # Convert to a NumPy array
    image_array = image.numpy()

    # Get the middle slices for each view
    axial_slice = image_array[:, :, image_array.shape[2] // 2]  # Middle slice along Z-axis
    coronal_slice = image_array[:, image_array.shape[1] // 2, :]  # Middle slice along Y-axis
    sagittal_slice = image_array[image_array.shape[0] // 2, :, :]  # Middle slice along X-axis

    # Plot the slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial view (looking from top-down)
    axes[0].imshow(axial_slice, cmap="gray")
    axes[0].set_title("Axial View (Top-Down)")
    axes[0].axis("off")

    # Coronal view (looking from front)
    axes[1].imshow(coronal_slice, cmap="gray")
    axes[1].set_title("Coronal View (Front)")
    axes[1].axis("off")

    # Sagittal view (looking from the side)
    axes[2].imshow(sagittal_slice, cmap="gray")
    axes[2].set_title("Sagittal View (Side)")
    axes[2].axis("off")

    plt.show()

def inspect_nifti_file_nib(data_file):
    # Load the NIfTI file
    nii_img = nib.load(data_file)

    # Get the affine transformation matrix
    #affine = nii_img.affine
    #print("Affine Matrix:\n", affine)

    # Check header metadata
    header = nii_img.header
    #print("Header Info:\n", header)

    # Convert to a NumPy array
    image_array = nii_img.get_fdata(dtype=np.float32)
    print(f"dtype: {image_array.dtype}, min: {image_array.min()}, max: {image_array.max()}, shape: {image_array.shape}")

    # Get the middle slices for each view
    axial_slice = image_array[:, :, image_array.shape[2] // 2]  # Middle slice along Z-axis
    coronal_slice = image_array[:, image_array.shape[1] // 2, :]  # Middle slice along Y-axis
    sagittal_slice = image_array[image_array.shape[0] // 2, :, :]  # Middle slice along X-axis

    # Plot the slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial view (looking from top-down)
    axes[0].imshow(axial_slice, cmap="gray")
    axes[0].set_title("Axial View (Top-Down)")
    axes[0].axis("off")

    # Coronal view (looking from front)
    axes[1].imshow(coronal_slice, cmap="gray")
    axes[1].set_title("Coronal View (Front)")
    axes[1].axis("off")

    # Sagittal view (looking from the side)
    axes[2].imshow(sagittal_slice, cmap="gray")
    axes[2].set_title("Sagittal View (Side)")
    axes[2].axis("off")

    plt.show()

import nibabel as nib
import numpy as np

def get_nifti_header_info(data_file):
    print(f"file: {os.path.basename(data_file)}")
    # Load the NIfTI file
    nii_img = nib.load(data_file)

    # Get the header information
    header = nii_img.header

    # Get dimensions (D, H, W) -- assuming a 3D image
    dim = header.get_data_shape()
    print(f"Image dimensions: {dim}")

    # Get voxel size (spacing in mm along each axis)
    voxel_size = header.get_zooms()
    print(f"Voxel size (mm): {voxel_size}")

    # Get data type stored in the file
    data_type = header.get_data_dtype()
    print(f"Stored data type: {data_type}")

    # Load image data
    data = nii_img.get_fdata()

    # Compute statistics
    nan_count = np.isnan(data).sum()
    total_count = data.size

    print(f"Data min: {np.nanmin(data)}, max: {np.nanmax(data)}, mean: {np.nanmean(data)}")
    print(f"Number of NaN values: {nan_count} / {total_count}")


def remove_small_values(in_file, out_file, threshold=1.0):
    img = nib.load(in_file)
    data = img.get_fdata()

    # Threshold tiny background noise
    data[data < threshold] = 0  

    # Cast to int16 to match original
    data = data.astype(np.int16)

    # Save with original-style header
    new_img = nib.Nifti1Image(data, img.affine)
    new_img.header.set_data_dtype(np.int16)
    nib.save(new_img, out_file)

get_nifti_header_info('/mnt/d/dimes/dataset/ds004199t_fsl/sub-00001/t1_reg.nii.gz')
get_nifti_header_info('/mnt/d/dimes/dataset/ds004199t_fsl/sub-00001/flair_reg.nii.gz')
get_nifti_header_info('/mnt/d/dimes/dataset/ds004199t_fsl/sub-00001/gt_reg.nii.gz')
get_nifti_header_info('/mnt/d/dimes/dataset/ds004199t_fsl/sub-00001/thickness_reg.nii.gz')
get_nifti_header_info('/mnt/d/dimes/test/doughnut_thickness.nii.gz')
get_nifti_header_info('/mnt/d/dimes/test/doughnut_contrast.nii.gz')

#remove_small_values('/mnt/d/dimes/dataset/orig_dataset/sub-00001/t1_reg.nii.gz', '/mnt/d/dimes/dataset/orig_dataset/sub-00001/t1_reg2.nii.gz', threshold=128)
#inspect_nifti_file_nib('/mnt/d/dimes/dataset/ds004199/sub-00001/anat/sub-00001_acq-iso08_T1w.nii.gz')
#inspect_nifti_file_nib('/mnt/d/dimes/dataset/orig_dataset/sub-00001/t1_reg.nii.gz')
#inspect_nifti_file_nib('/mnt/d/dimes/dataset/freesurfer_dataset/test/sub-00001/t1_reg.nii.gz')
#inspect_nifti_file_nib('/mnt/d/dimes/dataset/freesurfer_dataset/test/sub-00001/thickness.nii.gz')
#inspect_nifti_file_nib('/mnt/d/dimes/dataset/freesurfer_dataset_fsl/test/sub-00001/thickness_reg.nii.gz')

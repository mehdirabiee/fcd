import glob
import os
import nibabel as nib
import numpy as np


def get_data(data_dir, params):
    # Get the paths for T1 and FLAIR images
    t1_files = sorted(glob.glob(os.path.join(data_dir, "**/t1_reg.nii.gz"), recursive=True))
    fl_files = sorted(glob.glob(os.path.join(data_dir, "**/flair_reg.nii.gz"), recursive=True))

    # Initialize an empty list to store the data dictionaries
    data_dict = []

    # Iterate over pairs of T1 and FLAIR images
    for t1_f, fl_f in zip(t1_files, fl_files):
        # Initialize a dictionary to store the image and its corresponding label (if exists)
        data_entry = {'image': []}

        # Add the image(s) to the dictionary based on the sequence type
        if params['seq'] == 't1':
            data_entry['image'] = [t1_f]
        elif params['seq'] == 't2':
            data_entry['image'] = [fl_f]
        else:
            data_entry['image'] = [t1_f, fl_f]  # If using both, add both T1 and FLAIR

        # Check if the `gt_reg.nii.gz` segmentation file exists for this subject
        label_f = os.path.join(os.path.dirname(t1_f), "gt_reg.nii.gz")
        if os.path.exists(label_f):
            data_entry['label'] = label_f
            data_dict.append(data_entry)

        '''
        zero_mask_f = os.path.join(os.path.dirname(t1_f), "zero_mask.nii.gz")
        gt_f = label_f if os.path.exists(label_f) else zero_mask_f

        if os.path.exists(gt_f):
            data_entry['label'] = gt_f  
        else:
            zero_mask = np.zeros_like(nib.load(fl_f).get_fdata(), dtype=np.float32)
            zero_mask_img = nib.Nifti1Image(zero_mask, np.eye(4))  # Create a NIfTI image from zero mask
            nib.save(zero_mask_img, zero_mask_f)  # Save it to disk
            data_entry['label'] = zero_mask_f  # Save path of the generated zero mask
        # Append the dictionary to the data list
        data_dict.append(data_entry)
        '''

    return data_dict

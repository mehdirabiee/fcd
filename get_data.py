import glob
import os
import nibabel as nib
import numpy as np


def get_data(data_dir, params, include_thickness=False):
    """Get data paths for training and validation.
    
    Args:
        data_dir: Directory containing the preprocessed data
        params: Dictionary containing parameters like 'seq' for sequence type
        include_thickness: Whether to include thickness maps as additional channel
    
    Returns:
        List of dictionaries containing paths to images and labels
    """
    # Get the paths for T1 and FLAIR images
    t1_files = sorted(glob.glob(os.path.join(data_dir, "**/t1_reg.nii.gz"), recursive=True))
    fl_files = sorted(glob.glob(os.path.join(data_dir, "**/flair_reg.nii.gz"), recursive=True))
    
    # Initialize an empty list to store the data dictionaries
    data_dict = []

    # Iterate over pairs of T1 and FLAIR images
    for t1_f, fl_f in zip(t1_files, fl_files):
        # Initialize a dictionary to store the image and its corresponding label
        data_entry = {'image': []}

        # Add the image(s) to the dictionary based on the sequence type
        if params['seq'] == 't1':
            data_entry['image'] = [t1_f]
        elif params['seq'] == 't2':
            data_entry['image'] = [fl_f]
        else:
            # Default case: include both T1 and FLAIR
            data_entry['image'] = [t1_f, fl_f]

        # Add thickness map if requested
        if include_thickness:
            # Get subject ID from the path
            subject_id = os.path.basename(os.path.dirname(t1_f))
            
            # Find corresponding thickness map
            thickness_dir = os.path.join(os.path.dirname(data_dir), "thickness_maps")
            thickness_f = os.path.join(thickness_dir, subject_id, "thickness_CorticalThickness.nii.gz")
            
            # Skip if thickness map doesn't exist
            if not os.path.exists(thickness_f):
                print(f"Warning: Thickness map not found for subject {subject_id}")
                continue
                
            # Append thickness map as additional channel
            data_entry['image'].append(thickness_f)

        # Check if the `gt_reg.nii.gz` segmentation file exists for this subject
        label_f = os.path.join(os.path.dirname(t1_f), "gt_reg.nii.gz")
        if os.path.exists(label_f):
            data_entry['label'] = label_f
            data_dict.append(data_entry)

    return data_dict

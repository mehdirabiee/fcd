import os
import subprocess
import nibabel as nib
import numpy as np
from glob import glob

def extract_cortical_thickness(dataset_dir, output_dir):
    subjects = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
    os.makedirs(output_dir, exist_ok=True)
    
    for subject in subjects:
        subject_id = os.path.basename(subject)
        t1_image = os.path.join(subject, "t1_reg.nii.gz")
        flair_image = os.path.join(subject, "flair_reg.nii.gz")
        
        if not os.path.exists(t1_image) or not os.path.exists(flair_image):
            print(f"Skipping {subject_id}, missing T1 or FLAIR image.")
            continue
        
        # Set FreeSurfer subjects directory
        fs_subjects_dir = os.path.join(subject, "freesurfer")
        os.makedirs(fs_subjects_dir, exist_ok=True)
        
        # Run recon-all (FreeSurfer segmentation and cortical thickness analysis)
        recon_cmd = [
            "recon-all", "-s", subject_id,
            "-i", t1_image,
            "-all", "-sd", fs_subjects_dir
        ]
        
        print(f"Processing {subject_id}...")
        subprocess.run(recon_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Extract cortical thickness map
        thickness_file = os.path.join(fs_subjects_dir, subject_id, "surf", "lh.thickness")
        if not os.path.exists(thickness_file):
            print(f"Skipping {subject_id}, thickness file not found.")
            continue
        
        # Convert thickness file to .nii.gz format
        output_nii = os.path.join(output_dir, f"{subject_id}_cortical_thickness.nii.gz")
        mri_cmd = [
            "mris_convert", "--to-scanner", thickness_file, output_nii
        ]
        subprocess.run(mri_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(output_nii):
            print(f"Saved cortical thickness map for {subject_id} to {output_nii}")
        else:
            print(f"Failed to generate cortical thickness map for {subject_id}")

# Example usage
dataset_dir = "/mnt/d/dimes/dataset/ds004199_fsl/"  # Change this to your dataset path
output_dir = "/mnt/d/dimes/dataset/ds004199_freesurfer/"  # Directory to save output cortical thickness maps
extract_cortical_thickness(dataset_dir, output_dir)

import argparse
import os
import shutil
from glob import glob

def copy_thickness_maps(source_root: str, destination_root: str):
    """
    Copies thickness.nii.gz files from each subject folder in source_root
    to the corresponding anat folder in destination_root.

    Args:
        source_root (str): Path to the root directory containing subject folders with thickness.nii.gz.
        destination_root (str): Path to the BIDS-style root where files should be copied to.
    """
    source_files = glob(os.path.join(source_root, "*", "thickness.nii.gz"))

    for src_path in source_files:
        subject_id = os.path.basename(os.path.dirname(src_path))  # e.g., sub-01
        dest_dir = os.path.join(destination_root, subject_id, "anat")
        dest_path = os.path.join(dest_dir, "thickness.nii.gz")

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_path} → {dest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy thickness maps to BIDS-style anat folders.")
    parser.add_argument("--source", "-s", type=str, help="Root directory containing subject folders with thickness.nii.gz files.")
    parser.add_argument("--dest", "-d", type=str, help="BIDS-style root directory where files should be copied to.")
    
    args = parser.parse_args()
    
    copy_thickness_maps(args.source, args.dest)

import glob
import os
import shutil
import random
from pathlib import Path

def get_data(data_dir, params):
    """Get data paths for training and validation.
    
    Args:
        data_dir: Directory containing the preprocessed data
        params: Dictionary containing parameters like 'seq' for sequence type
    
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

        # Check if the `gt_reg.nii.gz` segmentation file exists for this subject
        label_f = os.path.join(os.path.dirname(t1_f), "gt_reg.nii.gz")
        if os.path.exists(label_f):
            data_entry['label'] = label_f
            data_dict.append(data_entry)

    return data_dict

def organize_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize data into train, validation, test, and unlabeled splits.
    
    Args:
        data_dir (str): Root directory containing the data
        train_ratio (float): Ratio of data for training (default: 0.7)
        val_ratio (float): Ratio of data for validation (default: 0.15)
        test_ratio (float): Ratio of data for testing (default: 0.15)
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing paths to each split
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create split directories
    splits = ['train', 'val', 'test', 'unlabeled']
    split_dirs = {}
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        split_dirs[split] = split_dir
    
    # Get all subject directories
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d not in splits]
    
    # Separate labeled and unlabeled data
    labeled_subjects = []
    unlabeled_subjects = []
    
    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        label_file = os.path.join(subject_dir, 'gt_reg.nii.gz')
        
        if os.path.exists(label_file):
            labeled_subjects.append(subject)
        else:
            unlabeled_subjects.append(subject)
    
    # Shuffle labeled subjects
    random.shuffle(labeled_subjects)
    
    # Calculate split sizes
    n_total = len(labeled_subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split subjects
    train_subjects = labeled_subjects[:n_train]
    val_subjects = labeled_subjects[n_train:n_train + n_val]
    test_subjects = labeled_subjects[n_train + n_val:]
    
    # Move subjects to appropriate directories
    def move_subjects(subjects, target_dir):
        for subject in subjects:
            src = os.path.join(data_dir, subject)
            dst = os.path.join(target_dir, subject)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
    
    # Move subjects to their respective directories
    move_subjects(train_subjects, split_dirs['train'])
    move_subjects(val_subjects, split_dirs['val'])
    move_subjects(test_subjects, split_dirs['test'])
    move_subjects(unlabeled_subjects, split_dirs['unlabeled'])
    
    # Create a summary dictionary
    summary = {
        'total_subjects': len(subjects),
        'labeled_subjects': len(labeled_subjects),
        'unlabeled_subjects': len(unlabeled_subjects),
        'train_subjects': len(train_subjects),
        'val_subjects': len(val_subjects),
        'test_subjects': len(test_subjects),
        'directories': split_dirs
    }
    
    # Print summary
    print("\nData Organization Summary:")
    print(f"Total subjects: {summary['total_subjects']}")
    print(f"Labeled subjects: {summary['labeled_subjects']}")
    print(f"Unlabeled subjects: {summary['unlabeled_subjects']}")
    print(f"Training subjects: {summary['train_subjects']}")
    print(f"Validation subjects: {summary['val_subjects']}")
    print(f"Testing subjects: {summary['test_subjects']}")
    print("\nData organized in directories:")
    for split, dir_path in split_dirs.items():
        print(f"{split}: {dir_path}")
    
    return summary

# Example usage:
if __name__ == "__main__":
    data_dir = "/mnt/d/dimes/dataset/ds004199_fsl/"
    summary = organize_data_splits(
        data_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

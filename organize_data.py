import argparse
import os
import shutil
import random
import nibabel as nib
from sklearn.model_selection import KFold

def deorganize_data(data_dir):
    split_dirs = [os.path.join(data_dir, d) for d in ['train', 'val', 'test', 'unlabeled']]
    subject_dirs = []
    for split_dir in split_dirs:
        if os.path.exists(split_dir):
            subject_dirs.extend(
                [os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            )
    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)
        dst = os.path.join(data_dir, subject_name)
        shutil.move(subject_dir, dst)

def save_split_assignments(split_dict, output_file):
    with open(output_file, "w") as f:
        for split, subjects in split_dict.items():
            for subj in sorted(subjects):
                f.write(f"{subj} {split}\n")

def load_split_assignments(assignments_file):
    if not os.path.exists(assignments_file):
        raise FileNotFoundError(f"Split assignment file not found: {assignments_file}")
    split_dict = {'train': [], 'val': [], 'test': []}
    with open(assignments_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject, split = parts
                if split in split_dict:
                    split_dict[split].append(subject)
    return split_dict

def organize_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                         seed=42, assignments_file=None):
    random.seed(seed)
    splits = ['train', 'val', 'test', 'unlabeled']
    split_dirs = {split: os.path.join(data_dir, split) for split in splits}
    for split_dir in split_dirs.values():
        os.makedirs(split_dir, exist_ok=True)

    all_subjects = [d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d)) and d not in splits]

    labeled, unlabeled = [], []
    for subject in all_subjects:
        label_file = os.path.join(data_dir, subject, 'gt_reg.nii.gz')
        if os.path.exists(label_file):
            label_data = nib.load(label_file).get_fdata()
            (labeled if label_data.sum() > 0 else unlabeled).append(subject)
        else:
            unlabeled.append(subject)

    labeled = sorted(labeled)
    unlabeled = sorted(unlabeled)

    if assignments_file:
        split_lists = load_split_assignments(assignments_file)
        train_subjects = [s for s in split_lists.get('train', []) if s in labeled]
        val_subjects = [s for s in split_lists.get('val', []) if s in labeled]
        test_subjects = [s for s in split_lists.get('test', []) if s in labeled]
    else:
        random.shuffle(labeled)
        n_total = len(labeled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        train_subjects = labeled[:n_train]
        val_subjects = labeled[n_train:n_train + n_val]
        test_subjects = labeled[n_train + n_val:]

    def move_subjects(subjects, target_dir):
        for subject in subjects:
            src = os.path.join(data_dir, subject)
            dst = os.path.join(target_dir, subject)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)

    move_subjects(train_subjects, split_dirs['train'])
    move_subjects(val_subjects, split_dirs['val'])
    move_subjects(test_subjects, split_dirs['test'])
    move_subjects(unlabeled, split_dirs['unlabeled'])

    split_dict = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects,
        'unlabeled': unlabeled
    }

    # Save a single file with subject and split assignment
    assignment_path = os.path.join(data_dir, "split_assignments.txt")
    save_split_assignments(split_dict, assignment_path)

    summary = {
        'total_subjects': len(all_subjects),
        'labeled_subjects': len(labeled),
        'unlabeled_subjects': len(unlabeled),
        'train_subjects': len(train_subjects),
        'val_subjects': len(val_subjects),
        'test_subjects': len(test_subjects),
        'directories': split_dirs
    }

    print("\nData Organization Summary:")
    for key, value in summary.items():
        if key == "directories":
            print("Split directories:")
            for k, v in value.items():
                print(f"{k}: {v}")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    print(f"\nSplit assignments saved to: {assignment_path}")
    return summary

def get_all_subjects(data_dir):
    """Return a sorted list of all valid subjects in data_dir"""
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subjects = sorted(subjects)
    return subjects


def create_kfold_splits(data_dir, k=5, val_fraction=0.1, random_seed=42):
    """
    Create k-fold train/val/test splits from subjects in data_dir.
    10% of the training part is used as validation for each fold.
    
    Args:
        data_dir: root directory containing subject subfolders
        k: number of folds
        val_fraction: fraction of training subjects to use as validation
        random_seed: random seed for reproducibility
        
    Returns:
        List of dicts, each dict contains:
            {'train': [...], 'val': [...], 'test': [...]}
    """
    subjects = get_all_subjects(data_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    
    splits = []
    
    for train_idx, test_idx in kf.split(subjects):
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        # select val_fraction of train_subjects as validation
        n_val = max(1, int(len(train_subjects) * val_fraction))
        random.seed(random_seed)  # for reproducibility
        val_subjects = random.sample(train_subjects, n_val)
        
        # remove val subjects from training
        train_subjects_final = [s for s in train_subjects if s not in val_subjects]
        
        splits.append({
            'train': train_subjects_final,
            'val': val_subjects,
            'test': test_subjects
        })
    
    return splits


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Manage dataset splits for FCD detection.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Split ratios for simple train/val/test split
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    
    # Optional assignments file
    parser.add_argument("--assignments_file", type=str, help="Path to save or load split assignments")
    parser.add_argument("--output_dir", type=str, help="Directory to save k-fold split files")

    # Flags for functionality
    parser.add_argument("--deorganize", action="store_true", help="Deorganize the data only")
    parser.add_argument("--organize", action="store_true", help="Organize the data after deorganizing")
    parser.add_argument("--kfold", type=int, help="Create k-fold splits (specify number of folds)")
    parser.add_argument("--save_split_only", action="store_true", help="Save current split assignments without reorganizing")
    
    args = parser.parse_args()

    # ------------------------
    # Deorganize only
    # ------------------------
    if args.deorganize and not args.organize and not args.kfold and not args.save_split_only:
        deorganize_data(args.data_dir)
        print("Data deorganized.")
        exit(0)

    # ------------------------
    # Save current split only
    # ------------------------
    if args.save_split_only:
        split_dict = {split: [] for split in ['train', 'val', 'test', 'unlabeled']}
        for split in split_dict:
            split_dir = os.path.join(args.data_dir, split)
            if os.path.exists(split_dir):
                split_dict[split] = sorted([
                    d for d in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, d))
                ])
        assignment_path = args.assignments_file
        if not assignment_path:
            raise ValueError("Specify --assignments_file to save split assignments.")
        save_split_assignments(split_dict, assignment_path)
        print(f"Current split assignment saved to: {assignment_path}")
        exit(0)

    # ------------------------
    # Create k-fold splits only
    # ------------------------
    if args.kfold:
        if not args.output_dir:
            raise ValueError("Specify --output_dir to save k-fold split files")
        splits = create_kfold_splits(args.data_dir, k=args.kfold, val_fraction=args.val_ratio, random_seed=args.seed)
        os.makedirs(args.output_dir, exist_ok=True)
        for i, split in enumerate(splits):
            fold_file = os.path.join(args.output_dir, f"split{i+1}.txt")
            save_split_assignments(split, fold_file)
            print(f"Saved fold {i+1} assignments to: {fold_file}")
        exit(0)

    # ------------------------
    # Deorganize and organize
    # ------------------------
    if args.deorganize or args.organize:
        if args.deorganize:
            deorganize_data(args.data_dir)
            print("Data deorganized.")
        organize_data_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            assignments_file=args.assignments_file
        )
        print("Data organized according to specified ratios.")
        exit(0)

    # ------------------------
    # If none of the above flags are set
    # ------------------------
    print("No action specified. Use --help to see available options.")

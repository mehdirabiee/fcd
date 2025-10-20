import os
import glob


def get_data(data_dir, params, subjects_list=None):
    """Get data paths for training and validation.
    
    Args:
        data_dir: Directory containing the preprocessed data
        params: Dictionary containing parameters like 'seq' for sequence type.
                Example: params['seq'] = 't1_reg+flair_reg+curv+area'
        subjects_list: Optional list of subject folder names to include (e.g. ["sub-1", "sub-2"])
    
    Returns:
        List of dictionaries containing paths to images and labels
    """
    seq_files = params['seq'].split('+')
    ref_seq = seq_files[0]
    data_dict = []

    # If no subject list is given, take all folders in data_dir
    if subjects_list is None:
        subjects_list = sorted([d for d in os.listdir(data_dir) 
                                if os.path.isdir(os.path.join(data_dir, d))])

    for subj in subjects_list:
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_dir):
            print(f"Warning: subject directory {subj_dir} not found, skipping.")
            continue

        # Find the reference sequence for this subject (recursively)
        ref_matches = glob.glob(os.path.join(subj_dir, f"**/{ref_seq}.nii.gz"), recursive=True)
        if not ref_matches:
            print(f"Warning: {ref_seq}.nii.gz not found for {subj}, skipping.")
            continue

        # Take the directory where reference is found
        seq_dir = os.path.dirname(ref_matches[0])
        data_entry = {'image': []}
        valid = True

        # Add all requested sequences (must be in same folder as ref)
        for seq in seq_files:
            seq_path = os.path.join(seq_dir, f"{seq}.nii.gz")
            if os.path.exists(seq_path):
                data_entry['image'].append(seq_path)
            else:
                print(f"Warning: {seq_path} not found, skipping {subj}.")
                valid = False
                break

        if not valid:
            continue

        # Thickness
        if params.get('thickness', False):
            thickness_filename = params.get('thickness_filename', 'thickness')
            thickness = os.path.join(seq_dir, f"{thickness_filename}.nii.gz")
            if os.path.exists(thickness):
                data_entry['image'].append(thickness)

        # Ground truth
        label_f = os.path.join(seq_dir, "gt_reg.nii.gz")
        if os.path.exists(label_f):
            data_entry['label'] = label_f
            data_dict.append(data_entry)
        else:
            print(f"Warning: gt_reg.nii.gz not found for {subj}, skipping.")

    return data_dict

def read_split_file(list_file):
    """Read a split file and return a dict {split_name: [subjects]}.

    Args:
        list_file: Path to text file with lines: "<subject_name> <split_name>"

    Returns:
        Dictionary mapping lowercase split names to lists of subject names.
    """
    split_dict = {}
    with open(list_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed lines
            subj, split = parts
            split = split.lower()
            if split not in split_dict:
                split_dict[split] = []
            split_dict[split].append(subj)
    return split_dict


def get_split_data(data_dir, list_file, split_name, params):
    """Get dataset for a specific split using a subject list file.

    Args:
        data_dir: Directory containing the preprocessed data
        list_file: Path to text file with lines: "<subject_name> <split_name>"
        split_name: Which split to load ("train", "val", "test"), case-insensitive
        params: Dictionary with parameters like 'seq'

    Returns:
        List of dictionaries (same format as get_data)
    """
    split_dict = read_split_file(list_file)
    split_name = split_name.lower()

    subjects = split_dict.get(split_name, [])
    if not subjects:
        print(f"Warning: no subjects found for split '{split_name}' in {list_file}")

    return get_data(data_dir, params, subjects_list=subjects)

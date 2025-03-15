import os
import glob
import argparse
from multiprocessing import Pool
from tqdm import tqdm

def compute_thickness_ants(args):
    t1_path, output_dir = args
    subject_id = os.path.basename(os.path.dirname(t1_path))
    output_prefix = os.path.join(output_dir, subject_id, 'thickness_')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # ANTs cortical thickness pipeline command
    cmd = f"""
    antsCorticalThickness.sh \
        -d 3 \
        -a {t1_path} \
        -e ./inputs/MNI152_T1_1mm.nii.gz \
        -m ./inputs/MNI152_T1_1mm_brain_mask.nii.gz \
        -p ./inputs/priors/priors%d.nii.gz \
        -o {output_prefix} \
        -t 1
    """
    
    # Run the command if thickness map doesn't exist
    if not os.path.exists(f"{output_prefix}CorticalThickness.nii.gz"):
        os.system(cmd)
    
    return f"{output_prefix}CorticalThickness.nii.gz"

def process_dataset(data_dir, output_dir, num_workers=4):
    # Find all T1w images
    t1_files = sorted(glob.glob(os.path.join(data_dir, "**/t1_reg.nii.gz"), recursive=True))
    
    # Prepare arguments for parallel processing
    args_list = [(t1_path, output_dir) for t1_path in t1_files]
    
    # Process in parallel
    with Pool(num_workers) as pool:
        thickness_maps = list(tqdm(pool.imap(compute_thickness_ants, args_list), 
                                 total=len(args_list),
                                 desc="Computing thickness maps"))
    
    return thickness_maps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cortical thickness maps using ANTs")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing preprocessed MRI data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save thickness maps")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel processes")
    
    args = parser.parse_args()
    process_dataset(args.data_dir, args.output_dir, args.num_workers) 
import argparse
import glob
import os
import multiprocessing

from tqdm import tqdm

mni152_path = './inputs/MNI152_T1_1mm.nii.gz'

def preprocess_file_fsl(args):
    sub_t1_path, sub_fl_path, sub_gt_path, sub_thickness_path, sub_save_dir, apply_robust_fov, delete_intermediate_files = args

    os.makedirs(sub_save_dir, exist_ok=True)
    sub_t1_reg = os.path.join(sub_save_dir, 't1_reg.nii.gz')
    sub_fl_reg = os.path.join(sub_save_dir, 'flair_reg.nii.gz')
    sub_gt_reg = os.path.join(sub_save_dir, 'gt_reg.nii.gz') if sub_gt_path is not None and os.path.exists(sub_gt_path) else None
    sub_thickness_reg = os.path.join(sub_save_dir, 'thickness_reg.nii.gz') if sub_thickness_path is not None and os.path.exists(sub_thickness_path) else None
    if (
        os.path.exists(sub_t1_reg)
        and os.path.exists(sub_fl_reg)
        and (sub_gt_reg is None or os.path.exists(sub_gt_reg))
        and (sub_thickness_reg is None or os.path.exists(sub_thickness_reg))
    ):
        print(f"Preprocessing already done for {sub_t1_path}, skipping...")
    else:
        # ========================================================
        # origin to std
        sub_t1_std = os.path.join(sub_save_dir, 't1_std.nii.gz')
        mat_ori2std = os.path.join(sub_save_dir, 'ori2std.mat')
        strcmd = f'fslreorient2std -m {mat_ori2std} {sub_t1_path} {sub_t1_std}'
        print(f'{strcmd}\n')
        os.system(strcmd)
        if apply_robust_fov:
            # robust fov
            sub_t1_fov = os.path.join(sub_save_dir, 't1_fov.nii.gz')
            mat_fov2std = os.path.join(sub_save_dir, 'fov2std.mat')
            strcmd = f'robustfov -i {sub_t1_std} -r {sub_t1_fov} -m {mat_fov2std}'
            print(f'{strcmd}\n')
            os.system(strcmd)

            mat_fov2mni = os.path.join(sub_save_dir, 'fov2mni.mat')
            strcmd = f'flirt -in {sub_t1_fov} -ref {mni152_path} -out {sub_t1_reg} -omat {mat_fov2mni} -dof 12 -cost corratio  \
                    -bins 256 -interp trilinear \
                    -searchrx -90 90 -searchry -90 90 -searchrz -90 90'
            print(f'{strcmd}\n')
            os.system(strcmd)

            # convert_xfm -omat ${T1}_nonroi2roi.mat -inverse ${T1}_roi2nonroi.mat
            mat_std2fov = os.path.join(sub_save_dir, 'std2fov.mat')
            strcmd = f'convert_xfm -omat {mat_std2fov} -inverse {mat_fov2std}'
            print(f'{strcmd}\n')
            os.system(strcmd)

            mat_ori2fov = os.path.join(sub_save_dir, 'ori2fov.mat')
            strcmd = f'convert_xfm -omat {mat_ori2fov} -concat {mat_std2fov} {mat_ori2std}'
            print(f'{strcmd}\n')
            os.system(strcmd)

            mat_ori2mni = os.path.join(sub_save_dir, 'ori2mni.mat')
            strcmd = f'convert_xfm -omat {mat_ori2mni} -concat {mat_fov2mni} {mat_ori2fov}'
            print(f'{strcmd}\n')
            os.system(strcmd)
        else:
            mat_std2mni = os.path.join(sub_save_dir, 'std2mni.mat')
            strcmd = f'flirt -in {sub_t1_std} -ref {mni152_path} -out {sub_t1_reg} -omat {mat_std2mni} -dof 12 -cost corratio  \
                    -bins 256 -interp trilinear \
                    -searchrx -90 90 -searchry -90 90 -searchrz -90 90'
            print(f'{strcmd}\n')
            os.system(strcmd)

            mat_ori2mni = os.path.join(sub_save_dir, 'ori2mni.mat')
            strcmd = f'convert_xfm -omat {mat_ori2mni} -concat {mat_std2mni} {mat_ori2std}'
            print(f'{strcmd}\n')
            os.system(strcmd)

        strcmd = f'flirt -in {sub_t1_path} -ref {mni152_path} -out {sub_t1_reg} -init {mat_ori2mni} -interp trilinear -applyxfm'
        print(f'{strcmd}\n')
        os.system(strcmd)

        if sub_thickness_path is not None and os.path.exists(sub_thickness_path):
            strcmd = f'flirt -in {sub_thickness_path} -ref {sub_t1_reg} -out {sub_thickness_reg} -init {mat_ori2mni} -interp nearestneighbour -applyxfm'
            print(f'{strcmd}\n')
            os.system(strcmd)

        # ========================================================
        # reg flair to t1-original, then to mni152
        sub_fl_reg0 = os.path.join(sub_save_dir, 'flair_reg0.nii.gz')
        mat_fl2t1 = os.path.join(sub_save_dir, 'mat_fl2t1.mat')
        strcmd = f'flirt -in {sub_fl_path} -ref {sub_t1_path} -out {sub_fl_reg0} -omat {mat_fl2t1} -dof 6 -cost mutualinfo  \
                -bins 256 -interp trilinear \
                -searchrx -90 90 -searchry -90 90 -searchrz -90 90'
        print(f'{strcmd}\n')
        os.system(strcmd)

        # Combine FLAIR-to-T1 and T1-to-MNI to get final FLAIR-to-MNI transform
        mat_fl2mni = os.path.join(sub_save_dir, 'mat_fl2mni.mat')
        strcmd = f'convert_xfm -omat {mat_fl2mni} -concat {mat_ori2mni} {mat_fl2t1}'
        print(f'{strcmd}\n')
        os.system(strcmd)

        strcmd = f'flirt -in {sub_fl_path} -ref {sub_t1_reg} -out {sub_fl_reg} -init {mat_fl2mni} -interp trilinear -applyxfm'
        print(f'{strcmd}\n')
        os.system(strcmd)

        # ========================================================
        if sub_gt_path is not None and os.path.exists(sub_gt_path):
            strcmd = f'flirt -in {sub_gt_path} -ref {sub_t1_reg} -out {sub_gt_reg} -init {mat_fl2mni} -interp nearestneighbour -applyxfm'
            print(f'{strcmd}\n')
            os.system(strcmd)

    # Cleanup intermediate files if delete_intermediate_files=True
    if delete_intermediate_files:
        files_to_keep = {sub_t1_reg, sub_fl_reg, sub_gt_reg, sub_thickness_reg}  # Only these should be kept
        for file in os.listdir(sub_save_dir):
            file_path = os.path.join(sub_save_dir, file)
            if file_path not in files_to_keep and os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def preprocess_dataset_fsl(data_dir, save_dir, apply_robust_fov=True, delete_intermediate_files=True, num_workers=-1):
    os.makedirs(save_dir, exist_ok=True)
    sub_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    tasks = []
    for sub_name in sub_names:
        sub_dir = os.path.join(data_dir, sub_name)
        t1_files = glob.glob(os.path.join(sub_dir, "**/*T1w.nii.gz"), recursive=True) + \
                   glob.glob(os.path.join(sub_dir, "**/t1_reg.nii.gz"), recursive=True)
        fl_files = glob.glob(os.path.join(sub_dir, "**/*FLAIR.nii.gz"), recursive=True) + \
                   glob.glob(os.path.join(sub_dir, "**/flair_reg.nii.gz"), recursive=True)
        gt_files = glob.glob(os.path.join(sub_dir, "**/*FLAIR_roi.nii.gz"), recursive=True) + \
                   glob.glob(os.path.join(sub_dir, "**/gt_reg.nii.gz"), recursive=True)
        thickness_files = glob.glob(os.path.join(sub_dir, "**/thickness.nii.gz"), recursive=True)
        if not t1_files or not fl_files:
            continue
        sub_save_dir = os.path.join(save_dir, sub_name)
        os.makedirs(sub_save_dir, exist_ok=True)
        tasks.append((t1_files[0], fl_files[0], gt_files[0] if gt_files else None, thickness_files[0] if thickness_files else None, sub_save_dir, apply_robust_fov, delete_intermediate_files))
    
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, len(tasks))
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(preprocess_file_fsl, tasks), total=len(tasks)))

def preprocess_IDEAS_dataset_fsl(data_root, save_dir, apply_robust_fov=True, delete_intermediate_files=True, num_workers=-1):
    bids_dir = os.path.join(data_root, "bids")
    masks_dir = os.path.join(data_root, "masks")

    if not os.path.isdir(bids_dir) or not os.path.isdir(masks_dir):
        raise ValueError(f"'bids' or 'masks' directory not found in {data_root}")

    sub_dirs = [d for d in os.listdir(bids_dir) if os.path.isdir(os.path.join(bids_dir, d)) and d.startswith("sub-")]
    tasks = []

    for sub in sub_dirs:
        sub_id = sub.replace("sub-", "")
        sub_anat_dir = os.path.join(bids_dir, sub, "anat")

        t1_path = os.path.join(sub_anat_dir, f"sub-{sub_id}_T1w.nii.gz")
        fl_path = os.path.join(sub_anat_dir, f"sub-{sub_id}_FLAIR.nii.gz")
        mask_path = os.path.join(masks_dir, sub_id, f"{sub_id}_MaskInRawData.nii.gz")

        # Check required modalities
        if not os.path.exists(t1_path):
            print(f"Skipping sub-{sub_id}: missing T1W: {t1_path}")
            continue
        if not os.path.exists(fl_path):
            print(f"Skipping sub-{sub_id}: missing FLAIR: {fl_path}")
            continue


        sub_save_dir = os.path.join(save_dir, f"sub-{sub_id}")
        os.makedirs(sub_save_dir, exist_ok=True)

        sub_gt_path = mask_path if os.path.exists(mask_path) else None

        tasks.append((t1_path, fl_path, sub_gt_path, None, sub_save_dir, apply_robust_fov, delete_intermediate_files))

    print(f"num valid subjects : {len(tasks)}")
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, len(tasks))
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(preprocess_file_fsl, tasks), total=len(tasks)))


if __name__ == '__main__':

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Preprocess fMRI dataset using FSL.')

    # Define the arguments
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                        help='Path to the input dataset directory.')
    parser.add_argument('--save_dir', '-s', type=str, required=True,
                        help='Path to the directory where preprocessed data will be saved.')
    parser.add_argument('--template_path', '-t', type=str, default='./MNI152_T1_1mm.nii.gz',
                        help='Path to the template file path.')
    parser.add_argument('--keep_intermediate', '-k', action='store_true',
                        help='Keep intermediate files after preprocessing.')
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="Number of parallel workers for preprocessing.")
    parser.add_argument('--no_robust_fov', action='store_true',
                        help='dont apply robust FOV correction during preprocessing.')
    parser.add_argument('--ideas', action='store_true', help='Use IDEAS dataset format.')
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    mni152_path = args.template_path
    num_workers = args.num_workers
    keep_intermediate = args.keep_intermediate
    apply_robust_fov = not args.no_robust_fov
    if args.ideas:
        preprocess_IDEAS_dataset_fsl(data_root=data_dir, save_dir=save_dir, apply_robust_fov=apply_robust_fov, delete_intermediate_files=not keep_intermediate, num_workers=num_workers)
    else:
        preprocess_dataset_fsl(data_dir=data_dir, save_dir=save_dir, apply_robust_fov=apply_robust_fov, delete_intermediate_files=not keep_intermediate, num_workers=num_workers)

#run: python preprocess_data.py -d '/mnt/d/dimes/dataset/ds004199/' -s '/mnt/d/dimes/dataset/ds004199_fsl/' -t  './inputs/MNI152_T1_1mm.nii.gz'
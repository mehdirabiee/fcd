import argparse
import glob
import os
import multiprocessing

from tqdm import tqdm

mni152_path = './inputs/MNI152_T1_1mm.nii.gz'

def preprocess_file_fsl(args):
    sub_t1_path, sub_fl_path, sub_gt_path, sub_save_dir, delete_intermediate_files = args

    os.makedirs(sub_save_dir, exist_ok=True)
    # ========================================================
    # origin to std
    sub_t1_std = os.path.join(sub_save_dir, 't1_std.nii.gz')
    mat_ori2std = os.path.join(sub_save_dir, 'ori2std.mat')
    strcmd = 'fslreorient2std -m {} {} {}'.format(mat_ori2std, sub_t1_path, sub_t1_std)
    print(f'{strcmd}\n')
    os.system(strcmd)

    # robust fov
    sub_t1_fov = os.path.join(sub_save_dir, 't1_fov.nii.gz')
    mat_fov2std = os.path.join(sub_save_dir, 'fov2std.mat')
    strcmd = 'robustfov -i {} -r {} -m {}'.format(sub_t1_std, sub_t1_fov, mat_fov2std)
    print(f'{strcmd}\n')
    os.system(strcmd)

    sub_t1_reg = os.path.join(sub_save_dir, 't1_reg.nii.gz')
    mat_fov2mni = os.path.join(sub_save_dir, 'fov2mni.mat')
    strcmd = 'flirt -in {} -ref {} -out {} -omat {} -dof 12 -cost corratio  \
            -bins 256 -interp trilinear \
            -searchrx -90 90 -searchry -90 90 -searchrz -90 90'.format(sub_t1_fov, mni152_path, sub_t1_reg, mat_fov2mni)
    print(f'{strcmd}\n')
    os.system(strcmd)

    # convert_xfm -omat ${T1}_nonroi2roi.mat -inverse ${T1}_roi2nonroi.mat
    mat_std2fov = os.path.join(sub_save_dir, 'std2fov.mat')
    strcmd = 'convert_xfm -omat {} -inverse {}'.format(mat_std2fov, mat_fov2std)
    print(f'{strcmd}\n')
    os.system(strcmd)

    mat_ori2fov = os.path.join(sub_save_dir, 'ori2fov.mat')
    strcmd = 'convert_xfm -omat {} -concat {} {}'.format(mat_ori2fov, mat_std2fov, mat_ori2std)
    print(f'{strcmd}\n')
    os.system(strcmd)

    mat_ori2mni = os.path.join(sub_save_dir, 'ori2mni.mat')
    strcmd = 'convert_xfm -omat {} -concat {} {}'.format(mat_ori2mni, mat_fov2mni, mat_ori2fov)
    print(f'{strcmd}\n')
    os.system(strcmd)

    strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp trilinear -applyxfm'.format(sub_t1_path, mni152_path,
                                                                                        sub_t1_reg, mat_ori2mni)
    print(f'{strcmd}\n')
    os.system(strcmd)

    # ========================================================
    # reg flair to t1-original, then to mni152
    sub_fl_reg0 = os.path.join(sub_save_dir, 'flair_reg0.nii.gz')
    mat_fl2t1 = os.path.join(sub_save_dir, 'mat_fl2t1.mat')
    strcmd = 'flirt -in {} -ref {} -out {} -omat {} -dof 6 -cost mutualinfo  \
            -bins 256 -interp trilinear \
            -searchrx -90 90 -searchry -90 90 -searchrz -90 90'.format(sub_fl_path, sub_t1_path, sub_fl_reg0, mat_fl2t1)
    print(f'{strcmd}\n')
    os.system(strcmd)

    sub_fl_reg = os.path.join(sub_save_dir, 'flair_reg.nii.gz')
    strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp trilinear -applyxfm'.format(sub_fl_reg0, sub_t1_reg,
                                                                                        sub_fl_reg, mat_ori2mni)
    print(f'{strcmd}\n')
    os.system(strcmd)


    # ========================================================
    sub_gt_reg = None
    if sub_gt_path is not None and os.path.exists(sub_gt_path):
        sub_gt_reg0 = os.path.join(sub_save_dir, 'gt_reg0.nii.gz')
        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp nearestneighbour -applyxfm'.format(sub_gt_path, sub_t1_path,
                                                                                                   sub_gt_reg0, mat_fl2t1)
        print(f'{strcmd}\n')
        os.system(strcmd)

        sub_gt_reg = os.path.join(sub_save_dir, 'gt_reg.nii.gz')
        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp nearestneighbour -applyxfm'.format(sub_gt_reg0, sub_t1_reg,
                                                                                                   sub_gt_reg, mat_ori2mni)
        print(f'{strcmd}\n')
        os.system(strcmd)

    # Cleanup intermediate files if delete_intermediate_files=True
    if delete_intermediate_files:
        files_to_keep = {sub_t1_reg, sub_fl_reg, sub_gt_reg}  # Only these should be kept
        for file in os.listdir(sub_save_dir):
            file_path = os.path.join(sub_save_dir, file)
            if file_path not in files_to_keep and os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")



'''
def preprocess_dataset_fsl(data_dir, save_dir, delete_intermediate_files=True):
    os.makedirs(save_dir, exist_ok=True)
    sub_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    sub_count = len(sub_names)
    for i in tqdm(range(sub_count)):
        sub_name = sub_names[i]
        sub_dir = os.path.join(data_dir, sub_name)
        t1_files = sorted( glob.glob(os.path.join(sub_dir, "*/*T1w.nii.gz"), recursive=True))
        fl_files = sorted( glob.glob(os.path.join(sub_dir, "*/*FLAIR.nii.gz"), recursive=True))
        gt_files = sorted( glob.glob(os.path.join(sub_dir, "*/*FLAIR_roi.nii.gz"), recursive=True))
        if len(t1_files) == 0 or len(fl_files) == 0:
            continue
        sub_save_dir = os.path.join(save_dir, sub_name)
        os.makedirs(sub_save_dir, exist_ok=True)

        sub_t1_path = t1_files[0]
        sub_fl_path = fl_files[0]
        sub_gt_path = gt_files[0] if len(gt_files) > 0 else None
        preprocess_file_fsl(sub_t1_path, sub_fl_path, sub_gt_path, sub_save_dir, delete_intermediate_files)
'''

def preprocess_dataset_fsl(data_dir, save_dir, delete_intermediate_files=True):
    os.makedirs(save_dir, exist_ok=True)
    sub_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    tasks = []
    for sub_name in sub_names:
        sub_dir = os.path.join(data_dir, sub_name)
        t1_files = sorted(glob.glob(os.path.join(sub_dir, "*/*T1w.nii.gz"), recursive=True))
        fl_files = sorted(glob.glob(os.path.join(sub_dir, "*/*FLAIR.nii.gz"), recursive=True))
        gt_files = sorted(glob.glob(os.path.join(sub_dir, "*/*FLAIR_roi.nii.gz"), recursive=True))
        if not t1_files or not fl_files:
            continue
        sub_save_dir = os.path.join(save_dir, sub_name)
        os.makedirs(sub_save_dir, exist_ok=True)
        tasks.append((t1_files[0], fl_files[0], gt_files[0] if gt_files else None, sub_save_dir, delete_intermediate_files))
    
    num_workers = min(multiprocessing.cpu_count(), len(tasks))
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
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    mni152_path = args.template_path
    keep_intermediate = args.keep_intermediate

    preprocess_dataset_fsl(data_dir=data_dir, save_dir=save_dir, delete_intermediate_files=not keep_intermediate)

#run: python preprocess_data.py -d '/mnt/d/dimes/dataset/ds004199/' -s '/mnt/d/dimes/dataset/ds004199_fsl/' -t  './inputs/MNI152_T1_1mm.nii.gz'
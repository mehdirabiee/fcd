import os
import shutil
import argparse

def delete_folders(parent_dir, names_file):
    # Read folder names
    with open(names_file, 'r') as f:
        folder_names = [line.strip() for line in f if line.strip()]

    for folder_name in folder_names:
        folder_path = os.path.join(parent_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
            except Exception as e:
                print(f"Error deleting {folder_path}: {e}")
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete subdirectories listed in a text file.")
    parser.add_argument('--dir', required=True, help='Path to the parent directory containing subfolders.')
    parser.add_argument('--names', required=True, help='Path to the text file containing subfolder names.')

    args = parser.parse_args()

    delete_folders(args.dir, args.names)

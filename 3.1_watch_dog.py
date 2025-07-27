import os
import shutil
import time
from datetime import datetime
import filecmp

def copy_fold_if_changed(source_dir, destination_root, fold_num="all", interval_minutes=30):
    """
    Copies a specified fold directory periodically, but only if its contents have changed
    since the last copy.  Saves each copy with a timestamped filename.

    Args:
        source_dir (str): Path to the directory containing the fold_* directory.
        destination_root (str): Root directory where copies will be saved.
        fold_num (int): The fold number to copy (default: 1).
        interval_minutes (int): Time interval between checks in minutes (default: 30).
    """

    fold_dir = os.path.join(source_dir, f"fold_{fold_num}")
    if not os.path.isdir(fold_dir):
        print(f"Error: Fold directory not found: {fold_dir}")
        return

    os.makedirs(destination_root, exist_ok=True)  # Create destination root if it doesn't exist

    last_copied_dir = None
    counter = 1

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_dir = os.path.join(destination_root, f"fold_{fold_num}_copy_{timestamp}")

            if last_copied_dir is None or not are_dirs_equal(fold_dir, last_copied_dir):
                try:
                    shutil.copytree(fold_dir, destination_dir)
                    print(f"Copied {fold_dir} to {destination_dir} (Copy #{counter})")
                    last_copied_dir = destination_dir  # Update the last copied directory
                except Exception as e:
                    print(f"Error copying {fold_dir} to {destination_dir}: {e}")
            else:
                print(f"No changes detected in {fold_dir}, skipping copy (Check #{counter})")

            counter += 1
            time.sleep(interval_minutes * 60)  # Sleep for the specified interval
    except KeyboardInterrupt:
        print("Copying interrupted by user.")

def are_dirs_equal(dir1, dir2):
    """
    Compares two directories recursively to check if their contents are equal.
    """
    compare = filecmp.dircmp(dir1, dir2)
    if compare.left_only or compare.right_only or compare.diff_files or compare.funny_files:
        return False
    for sub_dir in compare.common_dirs:
        if not are_dirs_equal(os.path.join(dir1, sub_dir), os.path.join(dir2, sub_dir)):
            return False
    return True
    print(f"file equal")
    

if __name__ == "__main__":
    source_directory = r"../data_result/nnUNet_results/Dataset101_CPSegmentation/nnUNetTrainer__nnUNetPlans__2d/"  # Replace with the actual source directory
    destination_directory = r"../data_result/nnUNet_results/Dataset101_CPSegmentation/nnUNetTrainer__nnUNetPlans__2d/"  # Replace with the actual destination directory
    fold_to_copy = "all"  # The fold number to copy (default is 1)
    copy_interval = 20  # Interval between checks in minutes (default is 30)

    copy_fold_if_changed(source_directory, destination_directory, fold_to_copy, copy_interval)
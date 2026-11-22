"""
Sync Script for Jignesh Chat Agent Study Guides
Synchronizes files between the project workspace (d:/Jignesh/AB_MLE/M3_AML/Jignesh Chat Agent)
and the centralized root folder (d:/Jignesh/Jignesh Chat Agent).
"""

import os
import shutil
from datetime import datetime

SOURCE_DIR = r"d:\Jignesh\AB_MLE\M3_AML\Jignesh Chat Agent"
TARGET_DIR = r"d:\Jignesh\Jignesh Chat Agent"

def sync_study_guides():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    count = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        rel_path = os.path.relpath(root, SOURCE_DIR)
        dest_root = os.path.join(TARGET_DIR, rel_path)
        os.makedirs(dest_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_root, file)

            # Copy if file does not exist or modified time is newer
            if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"[SYNCED] {os.path.join(rel_path, file)}")
                count += 1

    print(f"\nSynchronization completed successfully. Total files updated/copied: {count}")

if __name__ == "__main__":
    sync_study_guides()

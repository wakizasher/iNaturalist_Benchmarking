"""
🔧 Batch CSV File Path Fixer
Fix missing \t (backslash) in ALL CSV files in extended_data folder

Author: Nikita Gavrilov
"""

import pandas as pd
import os
import glob
from datetime import datetime
from pathlib import Path


def fix_csv_file_paths(csv_file_path, backup_dir=None):
    """
    Fix file paths in a single CSV file

    Args:
        csv_file_path (str): Path to CSV file
        backup_dir (str): Directory to save backup (optional)

    Returns:
        dict: Results of the fix operation
    """
    try:
        # Load the CSV
        df = pd.read_csv(csv_file_path)
        original_count = len(df)

        # Check if file_path column exists
        if 'file_path' not in df.columns:
            return {
                'success': False,
                'error': 'No file_path column found',
                'original_count': 0,
                'fixed_count': 0
            }

        # Count files with the issue (actual tab character)
        issue_count = df['file_path'].str.contains("iNaturalist\test_images", na=False).sum()

        if issue_count == 0:
            return {
                'success': True,
                'error': None,
                'original_count': original_count,
                'fixed_count': 0,
                'message': 'No issues found - file already correct'
            }

        # Create backup if backup directory provided
        if backup_dir:
            backup_filename = f"backup_{os.path.basename(csv_file_path)}"
            backup_path = os.path.join(backup_dir, backup_filename)
            df.to_csv(backup_path, index=False)

        # Fix the missing backslash issue - replace actual tab character with backslash
        df['file_path'] = df['file_path'].str.replace(
            "iNaturalist\test_images",
            "iNaturalist\\test_images",
            regex=False
        )

        # Save the fixed CSV (overwrite original)
        df.to_csv(csv_file_path, index=False)

        return {
            'success': True,
            'error': None,
            'original_count': original_count,
            'fixed_count': issue_count,
            'message': f'Fixed {issue_count} file paths'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_count': 0,
            'fixed_count': 0
        }


def batch_fix_csv_files(folder_path, create_backup=True, preview_only=False):
    """
    Fix file paths in all CSV files in a folder

    Args:
        folder_path (str): Path to folder containing CSV files
        create_backup (bool): Whether to create backup files
        preview_only (bool): If True, only show what would be done

    Returns:
        dict: Summary of results
    """
    print("🔧 BATCH CSV FILE PATH FIXER")
    print("=" * 50)
    print(f"📁 Target folder: {folder_path}")
    print(f"🔒 Backup enabled: {create_backup}")
    print(f"👁️ Preview mode: {preview_only}")
    print()

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return {'success': False, 'error': 'Folder not found'}

    # Find all CSV files
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"🔍 Found {len(csv_files)} CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"   {i:2d}. {filename}")

    if len(csv_files) == 0:
        print("❌ No CSV files found in the folder!")
        return {'success': False, 'error': 'No CSV files found'}

    print()

    # Create backup directory if needed
    backup_dir = None
    if create_backup and not preview_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(folder_path, f"backups_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📁 Created backup directory: {backup_dir}")
        print()

    # Process each CSV file
    results = []
    total_files_processed = 0
    total_paths_fixed = 0
    failed_files = []

    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"🔄 Processing {i}/{len(csv_files)}: {filename}")

        if preview_only:
            # Preview mode - just check for issues
            try:
                df = pd.read_csv(csv_file)
                if 'file_path' in df.columns:
                    issue_count = df['file_path'].str.contains("iNaturalist\test_images", na=False).sum()
                    print(f"   📊 Found {issue_count} paths to fix in {len(df)} total rows")
                    if issue_count > 0:
                        # Show example of what would be fixed
                        sample_row = df[df['file_path'].str.contains("iNaturalist\test_images", na=False)].iloc[0]
                        old_path = sample_row['file_path']
                        new_path = old_path.replace("iNaturalist\test_images", "iNaturalist\\test_images")
                        print(f"   📝 Example fix:")
                        print(f"      Old: {old_path}")
                        print(f"      New: {new_path}")
                else:
                    print(f"   ⚠️ No 'file_path' column found")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            # Actually fix the file
            result = fix_csv_file_paths(csv_file, backup_dir)
            results.append({
                'filename': filename,
                'result': result
            })

            if result['success']:
                if result['fixed_count'] > 0:
                    print(f"   ✅ Fixed {result['fixed_count']} paths in {result['original_count']} total rows")
                    total_paths_fixed += result['fixed_count']
                else:
                    print(f"   ✅ {result['message']}")
                total_files_processed += 1
            else:
                print(f"   ❌ Failed: {result['error']}")
                failed_files.append(filename)

        print()

    # Summary
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 40)

    if preview_only:
        print("👁️ Preview mode - no files were modified")
        print("🔧 Run with preview_only=False to actually fix files")
    else:
        print(f"✅ Successfully processed: {total_files_processed}/{len(csv_files)} files")
        print(f"🔧 Total file paths fixed: {total_paths_fixed}")

        if failed_files:
            print(f"❌ Failed files: {len(failed_files)}")
            for failed_file in failed_files:
                print(f"   - {failed_file}")

        if create_backup and backup_dir:
            print(f"📁 Backups saved to: {backup_dir}")

    return {
        'success': True,
        'total_files': len(csv_files),
        'processed_files': total_files_processed if not preview_only else 0,
        'total_fixes': total_paths_fixed if not preview_only else 0,
        'failed_files': failed_files if not preview_only else [],
        'backup_dir': backup_dir if not preview_only else None
    }


def main():
    """
    Main function to run the batch CSV fixer
    """
    # UPDATE THIS PATH TO YOUR ACTUAL EXTENDED_DATA FOLDER
    extended_data_folder = r"C:\Users\nikit\Pycharm\iNaturalist_Benchmarking\extended_data"

    print("🚀 BATCH CSV FILE PATH FIXER")
    print("=" * 60)
    print("This script will fix missing \\t (backslash) in ALL CSV files")
    print("in your extended_data folder")
    print()

    # First, run in preview mode to see what would be done
    print("👁️ PREVIEW MODE - Checking what needs to be fixed...")
    print("=" * 60)

    preview_results = batch_fix_csv_files(
        folder_path=extended_data_folder,
        create_backup=True,
        preview_only=True
    )

    if not preview_results['success']:
        print("❌ Preview failed. Please check the folder path and try again.")
        return

    print()
    print("🤔 Do you want to proceed with fixing all CSV files?")
    print("   This will:")
    print("   ✅ Create backups of all original files")
    print("   ✅ Fix the missing backslash issue in file paths")
    print("   ✅ Overwrite the original CSV files with fixed versions")
    print()

    # Get user confirmation
    response = input("Proceed? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        print("\n🔧 FIXING MODE - Actually fixing the files...")
        print("=" * 50)

        fix_results = batch_fix_csv_files(
            folder_path=extended_data_folder,
            create_backup=True,
            preview_only=False
        )

        if fix_results['success']:
            print("\n🎉 BATCH FIXING COMPLETED SUCCESSFULLY!")
            print(f"✅ Processed {fix_results['processed_files']} files")
            print(f"🔧 Fixed {fix_results['total_fixes']} file paths")

            if fix_results['backup_dir']:
                print(f"📁 Backups saved to: {fix_results['backup_dir']}")

            print("\n💡 You can now use any of the fixed CSV files in your experiments!")
        else:
            print("❌ Batch fixing failed!")

    else:
        print("🔒 Operation cancelled. No files were modified.")


# Additional utility functions
def restore_from_backup(backup_folder, target_folder):
    """
    Restore CSV files from backup folder

    Args:
        backup_folder (str): Path to backup folder
        target_folder (str): Path to target folder to restore to
    """
    print(f"🔄 Restoring CSV files from {backup_folder} to {target_folder}")

    backup_files = glob.glob(os.path.join(backup_folder, "backup_*.csv"))
    restored_count = 0

    for backup_file in backup_files:
        # Extract original filename
        backup_filename = os.path.basename(backup_file)
        original_filename = backup_filename.replace("backup_", "")
        target_path = os.path.join(target_folder, original_filename)

        try:
            # Copy backup to original location
            import shutil
            shutil.copy2(backup_file, target_path)
            print(f"   ✅ Restored: {original_filename}")
            restored_count += 1
        except Exception as e:
            print(f"   ❌ Failed to restore {original_filename}: {e}")

    print(f"📊 Restored {restored_count}/{len(backup_files)} files")


def check_all_csvs_status(folder_path):
    """
    Check the status of all CSV files in folder (fixed or not)

    Args:
        folder_path (str): Path to folder containing CSV files
    """
    print(f"🔍 Checking status of all CSV files in: {folder_path}")
    print("=" * 60)

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)

        try:
            df = pd.read_csv(csv_file)
            if 'file_path' in df.columns:
                issue_count = df['file_path'].str.contains("iNaturalist est_images", na=False).sum()
                status = "❌ NEEDS FIXING" if issue_count > 0 else "✅ FIXED"
                print(f"{status} - {filename} ({issue_count} issues in {len(df)} rows)")
            else:
                print(f"⚠️ NO FILE_PATH - {filename}")
        except Exception as e:
            print(f"❌ ERROR - {filename}: {e}")


if __name__ == "__main__":
    main()
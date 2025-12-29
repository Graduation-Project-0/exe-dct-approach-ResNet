import os
import shutil
import glob
from pathlib import Path

def setup_directories(base_dir):
    targets = {
        'benign': os.path.join(base_dir, 'benign'),
        'malware': os.path.join(base_dir, 'malware')
    }
    
    for path in targets.values():
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory exists: {path}")
        
    return targets

def move_files(source_patterns, target_dir, dry_run=False):
    moved_count = 0
    errors = 0
    
    for pattern in source_patterns:
        files = glob.glob(pattern, recursive=True)
        
        for src_path in files:
            if not os.path.isfile(src_path):
                continue
                
            file_name = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, file_name)
            
            # Avoid moving file to itself
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue

            # Handle name collisions
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(dst_path):
                    new_name = f"{base}_{counter}{ext}"
                    dst_path = os.path.join(target_dir, new_name)
                    counter += 1
            
            try:
                if dry_run:
                    print(f"[DRY RUN] Move: {src_path} -> {dst_path}")
                else:
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {src_path} -> {dst_path}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {src_path}: {e}")
                errors += 1
                
    return moved_count, errors

def cleanup_empty_dirs(paths):
    for path in paths:
        if os.path.exists(path) and os.path.isdir(path):
            if not os.listdir(path):
                try:
                    os.rmdir(path)
                    print(f"Removed empty directory: {path}")
                except OSError:
                    pass
            else:
                 pass

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    print(f"Data Root: {data_dir}")
    
    confirm = input("This will move files to reorganize the data folder. Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    targets = setup_directories(data_dir)
    
    benign_sources = [
        os.path.join(data_dir, 'Benign', '**', '*'),
        os.path.join(data_dir, 'benign_test', '**', '*'),  # potential other naming
        os.path.join(data_dir, 'benign_train', '**', '*')
    ]
    
    malware_sources = [
        os.path.join(data_dir, 'Virus', '**', '*'),
        os.path.join(data_dir, 'virus', '**', '*'),
        os.path.join(data_dir, 'Malware', '**', '*')
    ]
    
    print("\nProcessing Benign Files...")
    b_count, b_err = move_files(benign_sources, targets['benign'])
    
    print("\nProcessing Malware Files...")
    m_count, m_err = move_files(malware_sources, targets['malware'])
    
    print(f"\nSummary:")
    print(f"Benign files moved: {b_count} (Errors: {b_err})")
    print(f"Malware files moved: {m_count} (Errors: {m_err})")
    
    print("\nNote: Source directories (like 'benign' or 'malware') were kept. You can manually delete them if empty.")

if __name__ == "__main__":
    main()  

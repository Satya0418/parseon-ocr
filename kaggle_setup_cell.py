# ══════════════════════════════════════════════════════════════════════════
# KAGGLE SETUP — Auto-detect OCR Dataset and Configure Paths
# ══════════════════════════════════════════════════════════════════════════
# Replace the first 2-3 cells of the Colab notebook with THIS SINGLE CELL
# ══════════════════════════════════════════════════════════════════════════

import os
import sys
import zipfile

print('═' * 80)
print('KAGGLE ENVIRONMENT DETECTION')
print('═' * 80)

# Step 1: Check if running on Kaggle
if not os.path.exists('/kaggle/input'):
    print('⚠ /kaggle/input not found — are you running on Kaggle?')
    print('  Go to Settings → Accelerator → GPU T4 x2')
    raise RuntimeError('Not running on Kaggle')

print('✓ Running on Kaggle')
print(f'✓ Working directory: {os.getcwd()}')

# Step 2: Deep scan to find ALL directories in /kaggle/input
print('\n📁 Scanning /kaggle/input/ for datasets...')

def scan_directory(path, max_depth=3, current_depth=0):
    """Recursively scan directory and return all paths."""
    all_paths = []
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                all_paths.append(item_path)
                if current_depth < max_depth:
                    all_paths.extend(scan_directory(item_path, max_depth, current_depth + 1))
    except PermissionError:
        pass
    return all_paths

# Get all directories under /kaggle/input
all_paths = scan_directory('/kaggle/input', max_depth=4)
print(f'Found {len(all_paths)} directories total')

# Check if /kaggle/input is empty
if len(all_paths) == 0:
    print('\n❌ No datasets found in /kaggle/input/')
    print('\n💡 TO FIX THIS:')
    print('  1. Look at the right sidebar → "Input" section')
    print('  2. Click "+ Add Input"')
    print('  3. Search for your uploaded dataset (e.g., "ocr-iam-dataset")')
    print('  4. Click "Add" to attach it to this notebook')
    print('  5. Re-run this cell')
    raise FileNotFoundError('No datasets attached to this notebook')

# Show top-level structure
top_level = sorted(os.listdir('/kaggle/input'))
print(f'\nTop-level folders in /kaggle/input/:')
for item in top_level:
    item_path = os.path.join('/kaggle/input', item)
    if os.path.isdir(item_path):
        contents = os.listdir(item_path)[:3]
        print(f'  📦 {item}/')
        for subitem in contents:
            subitem_path = os.path.join(item_path, subitem)
            if os.path.isdir(subitem_path):
                print(f'      📁 {subitem}/')
            else:
                print(f'      📄 {subitem}')

# Step 3: Search ALL paths for the OCR project
print('\n🔍 Searching for OCR project (looking for backend/ folder)...')

PROJECT_PATH = None
DATASET_NAME = None

# Search through all discovered paths
for path in all_paths:
    # Check if this path contains a 'backend' folder
    if os.path.isdir(os.path.join(path, 'backend')):
        # Verify it also has 'data' folder (to confirm it's our project)
        if os.path.isdir(os.path.join(path, 'data')):
            PROJECT_PATH = path
            # Extract dataset name (first folder after /kaggle/input/)
            relative = path.replace('/kaggle/input/', '')
            DATASET_NAME = relative.split('/')[0] if '/' in relative else relative
            print(f'  ✓ Found project at: {PROJECT_PATH}')
            print(f'  ✓ Dataset name: {DATASET_NAME}')
            break
    
    # Also check if 'backend' is directly in this path's name (in case path IS backend)
    if path.endswith('backend') and os.path.isdir(path):
        parent = os.path.dirname(path)
        if os.path.isdir(os.path.join(parent, 'data')):
            PROJECT_PATH = parent
            relative = parent.replace('/kaggle/input/', '')
            DATASET_NAME = relative.split('/')[0] if '/' in relative else relative
            print(f'  ✓ Found project at: {PROJECT_PATH}')
            print(f'  ✓ Dataset name: {DATASET_NAME}')
            break

# If still not found, try looking for zip files
if PROJECT_PATH is None:
    print('\n  Backend folder not found, looking for zip files...')
    for path in all_paths:
        try:
            for item in os.listdir(path):
                if item.endswith('.zip'):
                    zip_path = os.path.join(path, item)
                    print(f'  📦 Found zip: {zip_path}')
                    print(f'  ⏳ Extracting to /kaggle/working/ ...')
                    
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall('/kaggle/working')
                    
                    # Check what was extracted
                    for extracted_item in os.listdir('/kaggle/working'):
                        extracted_path = os.path.join('/kaggle/working', extracted_item)
                        if os.path.isdir(extracted_path):
                            if os.path.isdir(os.path.join(extracted_path, 'backend')):
                                PROJECT_PATH = extracted_path
                                DATASET_NAME = 'extracted'
                                print(f'  ✓ Extracted to: {PROJECT_PATH}')
                                break
                    if PROJECT_PATH:
                        break
        except (PermissionError, OSError):
            continue
        if PROJECT_PATH:
            break

# Step 4: Validate the project structure
if PROJECT_PATH is None:
    print('\n❌ Could not find OCR project in any location!')
    print('\nDEBUGGING INFO:')
    print('Expected to find a directory containing both:')
    print('  - backend/ folder')
    print('  - data/ folder')
    print('\nAll paths searched:')
    for p in all_paths[:20]:  # Show first 20 paths
        has_backend = '✓ backend' if os.path.isdir(os.path.join(p, 'backend')) else ''
        has_data = '✓ data' if os.path.isdir(os.path.join(p, 'data')) else ''
        markers = f' ({has_backend} {has_data})'.strip()
        print(f'  {p}{markers}')
    if len(all_paths) > 20:
        print(f'  ... and {len(all_paths) - 20} more')
    
    print('\n💡 SOLUTION:')
    print('  1. Check the "Input" panel on the right →')
    print('  2. Make sure you clicked "+ Add Input" and added your dataset')
    print('  3. The dataset should show "ocr-iam-dataset" or similar')
    print('  4. Try clicking the dataset name to browse its contents')
    raise FileNotFoundError('OCR project structure not found')

print('\n✅ PROJECT DETECTED SUCCESSFULLY!')
print(f'   Dataset name : {DATASET_NAME}')
print(f'   Source path  : {PROJECT_PATH}')
print(f'   (Will be copied to /kaggle/working/ for editing)')

# Step 5: Copy project to writable location (Kaggle input is READ-ONLY!)
print('\n⚠️  IMPORTANT: /kaggle/input/ is READ-ONLY')
print('   Copying project to /kaggle/working/ for patching...')

WORKING_PROJECT = '/kaggle/working/ocr_project'

# Check if already copied
if os.path.isdir(WORKING_PROJECT) and os.path.isdir(os.path.join(WORKING_PROJECT, 'backend')):
    print(f'   ✓ Project already exists in /kaggle/working/')
else:
    import shutil
    print(f'   📋 Copying from: {PROJECT_PATH}')
    print(f'   📋 Copying to:   {WORKING_PROJECT}')
    
    # Copy entire project
    if os.path.exists(WORKING_PROJECT):
        shutil.rmtree(WORKING_PROJECT)
    shutil.copytree(PROJECT_PATH, WORKING_PROJECT)
    print(f'   ✓ Copy complete!')

# Update PROJECT_PATH to the writable copy
PROJECT_PATH = WORKING_PROJECT

# Step 6: Verify required folders exist in writable copy
required_folders = ['backend', 'data']
print('\n📂 Verifying writable copy structure:')
for folder in required_folders:
    folder_path = os.path.join(PROJECT_PATH, folder)
    exists = os.path.isdir(folder_path)
    status = '✓' if exists else '✗'
    print(f'  {status} {folder}/')
    if exists and folder == 'data':
        # Check for IAM dataset
        iam_path = os.path.join(folder_path, 'iam_words')
        if os.path.isdir(iam_path):
            print(f'      ✓ iam_words/ found')
        else:
            print(f'      ⚠ iam_words/ not found (may need to download)')

# Step 7: Add project to Python path and set working directory
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
    print(f'\n✓ Added to sys.path: {PROJECT_PATH}')

os.chdir(PROJECT_PATH)
print(f'✓ Changed working directory to: {os.getcwd()}')

# Step 7: Show final structure
print('\n📋 Current directory contents:')
for item in sorted(os.listdir('.')):
    if os.path.isdir(item):
        print(f'   📁 {item}/')
    else:
        print(f'   📄 {item}')

print('\n' + '═' * 80)
print('✅ SETUP COMPLETE — Ready to run training cells!')
print('═' * 80)
print(f'\n📍 Working from: /kaggle/working/ocr_project/')
print('   (Writable copy - patches will be applied here)')
print('\nNext steps:')
print('  1. Skip the "Mount Google Drive" cells (Colab-specific)')
print('  2. Run "Install Dependencies" cell')
print('  3. Continue with data verification and training')
print('  4. Run the "CRITICAL FIX" patching cell before building the model')


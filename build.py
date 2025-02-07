import os
import shutil
import subprocess
import sys

def clean_build():
    """Clean build directories"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
def build_exe():
    """Build the executable using PyInstaller"""
    try:
        # Clean previous builds
        clean_build()
        
        # Run PyInstaller
        subprocess.run(['pyinstaller', 'tilin_scanner.spec'], check=True)
        
        print("Build completed successfully!")
        print("Executable can be found in the 'dist' directory")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(build_exe())

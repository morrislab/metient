from setuptools import setup, find_packages
import os
import shutil
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop

def install_projectppm():
    """Clone and build projectppm"""
    try:
        # Get the directory where projectppm should be installed
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metient', 'lib')
        os.makedirs(lib_dir, exist_ok=True)
        
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to lib directory
            os.chdir(lib_dir)
            
            # Clone projectppm if it doesn't exist
            if not os.path.exists('projectppm'):
                print("Cloning projectppm repository...")
                subprocess.check_call(['git', 'clone', 'https://github.com/ethanumn/projectppm'])
            
            # Build projectppm
            os.chdir('projectppm')
            print("Building projectppm...")
            subprocess.check_call(['bash', 'make.sh'])
            print("projectppm installation completed successfully!")
            
        finally:
            # Always return to original directory
            os.chdir(original_dir)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during projectppm installation: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during projectppm installation: {e}")
        raise

class CustomInstall(install):
    def run(self):
        install_projectppm()
        install.run(self)

class CustomDevelop(develop):
    def run(self):
        # install_projectppm()
        develop.run(self)

requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
with open(requirements_path) as f:
    requirements = f.read().splitlines()
    #print(requirements)

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='metient',
    version='0.1.3.4.12',
    url="https://github.com/divyakoyy/metient.git",
    packages=['metient', 'metient.util', 'metient.lib'],
    install_requires=requirements,
    package_data={
        'metient.lib': ['projectppm/*'],  # Include all projectppm files
    },
    include_package_data=True,
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    }
)

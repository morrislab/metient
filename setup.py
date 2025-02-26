import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

def install_projectppm():
    """Install projectppm dependency"""
    try:
        # Get the directory where projection.py is located
        projection_dir = os.path.join(os.path.dirname(__file__), 'metient', 'lib')
        print(f"Installing projectppm dependency in {projection_dir}...")
        
        # Create lib directory if it doesn't exist
        os.makedirs(projection_dir, exist_ok=True)
        
        # Clone and build projectppm
        os.chdir(projection_dir)
        if not os.path.exists('projectppm'):
            print("Cloning projectppm repository...")
            subprocess.check_call(['git', 'clone', 'https://github.com/ethanumn/projectppm'])
            os.chdir('projectppm')
            print("Building projectppm...")
            subprocess.check_call(['bash', 'make.sh'])
            print("projectppm installation completed successfully!")
        else:
            print("projectppm already exists, skipping installation")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during projectppm installation: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Unexpected error during projectppm installation: {e}", file=sys.stderr)
        raise

class CustomInstall(install):
    def run(self):
        install.run(self)
        install_projectppm()

class CustomDevelop(develop):
    def run(self):
        develop.run(self)
        install_projectppm()

requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
with open(requirements_path) as f:
    requirements = f.read().splitlines()
    #print(requirements)

setup(
    name='metient',
    version='0.1.3.4.10',
    url="https://github.com/divyakoyy/metient.git",
    packages=['metient', 'metient.util', 'metient.lib'],
    install_requires=requirements,
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
)

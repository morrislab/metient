from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='metient',
    version='0.1.3.5.2',
    url="https://github.com/divyakoyy/metient.git",
    packages=find_packages(),  # automatically finds metient, metient.util, metient.lib
    install_requires=requirements,
    package_data={
        'metient.lib': [
            'projectppm/bin/*',   # include executable and shared library
        ],
    },
    include_package_data=True,
)

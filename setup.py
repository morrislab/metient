from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [("==").join(req.split("=")[:2]) for req in f.read().splitlines()]
    # print(requirements)

setup(name='metient', version='1.0.1.dev1', url="https://github.com/divyakoyy/metient.git", packages=find_packages(), install_requires=requirements,)

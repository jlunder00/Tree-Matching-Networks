#Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/
#setup.py - for formatting as a package
import os
from setuptools import setup, find_packages

requirements = []
with open(os.path.dirname(__file__) + "/requirements.txt", "r") as R:
    for line in R:
        package = line.strip()
        requirements.append(package)

setup(
    name="Tree_Matching_Networks",
    version='1.0.0',
    description="Tree Matching Networks implementation and Applications",
    author="Jason Lunder",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)

"""
cluster
Clusters class averages and generates relevant input for histogram_viz
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "Clusters class averages and generates relevant input for histogram_viz".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None


setup(
    # Self-descriptive entries which should always be present
    name='class_average_clustering',
    author='Ishan Taneja',
    author_email='itaneja@scripps.edu',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    install_requires=[
        "numpy>=1.19.5",
        "mrcfile>=1.3.0",
        "matplotlib>=3.3.2",
        "opencv-python>=4.5.3.56",
        "scipy>=1.6.3",
        "scikit-learn>=0.23.1",
        "scikit-image>=0.18.3",
        "imutils>=0.5.4",
        "joblib>=0.15.1",
        "seaborn>=0.11.0",
        "networkx>=2.5"],

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)

from setuptools import find_packages, setup

setup(
    name='Undectable_ai',  # Use a unique project name
    packages=find_packages(),
    version='0.1.0',
    description='Project to detect AI generated text',
    author='Tom',
    license='MIT',
    install_requires=[
        'tensorflow',
        'pandas',
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
    ],
)
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.2',
                     'tensorflow-probability',
                     'torch',
                     'torchvision',
                     'numpy',
                     'nltk',
                     'networkx',
                     'dm-tree',
                     'matplotlib']


PACKAGES = [package
            for package in find_packages() if
            package.startswith('indigo')]


setup(name='indigo',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=PACKAGES,
      description='Transformer-InDIGO')

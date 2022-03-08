from setuptools import setup
from setuptools import find_packages

setup(
   name='ensmic',
   version='1.0',
   description='An analysis on Ensemble Learning optimized Neural Network Classification',
   url='https://github.com/muellerdo/ensmic',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description="An analysis on Ensemble Learning optimized Neural Network Classification",
   long_description_content_type="text/markdown",
   packages=find_packages(),
   install_requires=['tensorflow>=2.4.0',
                     'aucmedi>=0.3.0',
                     'pandas>=1.1.4',
                     'pillow>=7.2.0',
                     'plotnine>=0.7.1',
                     'tqdm>=4.35.0'],
   classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps."]
)

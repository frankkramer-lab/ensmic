from setuptools import setup
from setuptools import find_packages

setup(
   name='covidxscan',
   version='0.2',
   description='COVID-19 Screening and Quantitative Assessment of X-Ray Images and CT Volumes using Deep Learning',
   url='https://github.com/muellerdo/covid-xscan',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description="COVID-19 Screening and Quantitative Assessment of X-Ray Images and CT Volumes using Deep Learning",
   long_description_content_type="text/markdown",
   packages=find_packages(),
   install_requires=['tensorflow==2.2.0',
                     'miscnn==0.37',
                     'Pillow'],
   classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps."]
)

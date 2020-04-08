from setuptools import setup
from setuptools import find_packages

setup(
   name='covidxscan',
   version='0.1',
   description='COVID-19 Detection of X-Ray Images using Deep Learning',
   url='https://github.com/muellerdo/covid-xscan',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description="COVID-19 Detection of X-Ray Images using Deep Learning",
   long_description_content_type="text/markdown",
   packages=find_packages(),
   install_requires=['pandas',
                     'tensorflow==2.1.0',
                     'miscnn',
                     'Pillow',
                     'pydicom'],
   classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps."]
)

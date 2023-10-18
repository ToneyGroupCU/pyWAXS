from setuptools import setup, find_packages
from distutils.core import setup
import sys,os
# sys.path.append(os.path.dirname(__file__))
setup()

# setup(
#     name='pyWAXS',
#     version='0.1.0',
#     description='pyWAXS is a Python-based X-ray scattering data reduction package.',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
#     author='Keith White',
#     author_email='keith.white@colorado.edu',
#     url='https://github.com/ToneyGroupCU/pyWAXS',
#     # packages=find_packages(),  # Removed 'where'
#     # packages=['main'],
#     packages=['pyWAXS'],  # The name you want to import
#     package_dir={'pyWAXS': 'main'},  # Directory where your actual modules are
#     python_requires='>=3.11',  # Ensure this Python version is correct
#     # install_requires=[
#     #     # Add dependencies here
#     # ],
#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: Science/Research',
#         'Programming Language :: Python :: 3.11',
#     ],
#     keywords=['WAXS', 'X-ray Scattering', 'Data Analysis'],
# )

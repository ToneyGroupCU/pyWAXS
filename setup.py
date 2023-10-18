from setuptools import setup, find_packages

setup(
    name='pyWAXS',
    version='0.1.0',
    description='pyWAXS is a Python-based X-ray scattering data reduction package.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Keith White',
    author_email='keith.white@colorado.edu',
    url='https://github.com/ToneyGroupCU/pyWAXS',
    packages=find_packages(where='./main'),
    package_dir={'': './main'},
    python_requires='>=3.11',
    # install_requires=[
    #     # Add dependencies here
    # ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.11',
    ],
    keywords=['WAXS', 'X-ray Scattering', 'Data Analysis'],
)

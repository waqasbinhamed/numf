from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='numf',
    version='1.0.0',
    description='Python implementation of Nonnegative Unimodal Matrix Factorization (NuMF)',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/waqasbinhamed/numf',
    author='Waqas Bin Hamed',
    author_email='waqasbinhamed@gmail.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

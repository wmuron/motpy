from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='motpy',
    version='0.0.9',
    url='https://github.com/wmuron/motpy.git',
    download_url='https://github.com/wmuron/motpy/releases/tag/v0.0.9-alpha',
    author='Wiktor Muron',
    author_email='wiktormuron@gmail.com',
    description='Library for track-by-detection multi object tracking implemented in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=("tests",)),
    python_requires='>3.6',
    install_requires=['numpy',
                      'scipy',
                      'filterpy'],
    license='MIT',
    keywords=['multi-object-tracking', 'object-tracking', 'kalman-filter'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License'
    ]
)

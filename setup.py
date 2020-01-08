from setuptools import setup, find_packages

setup(
    name='motpy',
    version='0.0.6',
    url='https://github.com/wmuron/motpy.git',
    download_url='https://github.com/wmuron/motpy/releases/tag/v0.0.6-alpha',
    author='Wiktor Muron',
    author_email='wiktormuron@gmail.com',
    description='Library for track-by-detection multi object tracking implemented in python',
    packages=find_packages(exclude=("tests",)),
    python_requires='>3.6',
    install_requires=['numpy',
                      'scipy',
                      'filterpy',
                      'loguru'],
    license='MIT',
    keywords=['multi-object-tracking', 'object-tracking', 'kalman-filter'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License'
    ]
)

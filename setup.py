#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='apprecommender',
    description="Package recommender for GNU packages",
    version='0.1',
    url='https://github.com/tassia/AppRecommender',
    author='Tassia Camoes Araujo',
    author_email='tassia@acaia.ca',
    license='GPLv3.txt',
    packages=find_packages(),
    setup_requires=['nose>=1.3', 'mock'],
    test_suite='nose.collector',
    package_data={
        '': [
            '*.sh', 'bin/*.sh'
        ]
    },
    entry_points={
        'console_scripts': [
            'apprec = bin.apprec:main',
        ]
    },
)

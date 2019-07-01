#!/usr/bin/env python

"""
distutils/setuptools install script.
"""
import re
from setuptools import find_packages, setup

VERSION_RE = re.compile(r'''([0-9dev.]+)''')


def get_version():
    with open('VERSION', 'r') as fh:
        init = fh.read().strip()
    return VERSION_RE.search(init).group(1)


def get_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name='grafuple',
    version=get_version(),
    description='Decompiled dotnet pagerank random forest classifier',
    author='Michael Slawinski',
    author_email='mslawinski@cylance.com',
    url='http://stash.d.cylance.com:7990/projects/LEARN/repos/grafuple',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={
        'grafuple': [
            'VERSION',
            'requirements.txt',
            'README.md',
        ]
    },
    include_package_data=True,
    install_requires=get_requirements(),
    license="",
)

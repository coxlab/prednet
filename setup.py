#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='prednet',
    use_scm_version={
        'local_scheme': 'dirty-tag',
        'write_to': 'src/prednet/_version.py',
        'fallback_version': '0.0.0',
    },
    license='MIT',
    description='Code and models accompanying Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning by Bill Lotter, Gabriel Kreiman, and David Cox. The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005).',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Bill Lotter',
    author_email='contact@coxlab.org',
    url='https://github.com/coxlab/prednet',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    # package_data={'prednet.tests.resources': ['black.mpg']},
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
    project_urls={
        'Documentation': 'https://prednet.readthedocs.io/',
        'Changelog': 'https://prednet.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/coxlab/prednet/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        'tensorflow-gpu>=1.13.1,<2.0',
        'Keras>=2.2.4,<2.4.0',
        # Upgrading TensorFlow, but not Keras, yields tensorflow.python.eager.core._FallbackException: This function does not handle the case of the path where all inputs are not already EagerTensors.
        # Upgrading Keras yields Upgrading keras yields ModuleNotFoundError: No module named 'keras.legacy'
        # 'tensorflow',
        # 'Keras',
        'requests',
        'bs4',
        'jinja2',
        'importlib_resources;python_version<"3.7"',
        'imageio',
        'imageio-ffmpeg',
        'scipy>=1.2.0',
        'pillow',
        # 'scikit-video @ git+https://github.com/dHannasch/scikit-video.git@branch-to-install',
        'scikit-video',
        'ffmpeg-python',
        'matplotlib',
        'jupyterlab',
        'hickle',
        'pytest',
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'pytest-runner',
        'setuptools_scm>=3.3.1',
    ],
    entry_points={
        'console_scripts': [
            'prednet = prednet.cli:main',
        ]
    },
)

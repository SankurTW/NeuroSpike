# setup.py
"""
NeuroSpike - Spiking Neural Network Framework
Setup configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name='neurospike',
    version='1.0.0',
    author='NeuroSpike Team',
    author_email='neurospike@example.com',
    description='A comprehensive Spiking Neural Network framework for event-based neuromorphic computing',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/neurospike',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ],
    },
    keywords='spiking neural networks, neuromorphic computing, event-based vision, STDP, LIF neurons',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/neurospike/issues',
        'Source': 'https://github.com/yourusername/neurospike',
        'Documentation': 'https://neurospike.readthedocs.io',
    },
)
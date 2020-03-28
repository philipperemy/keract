from setuptools import setup

from keract import __version__

setup(
    name='keract',
    version=__version__,
    description='Keract - Tensorflow Keras Activations and Gradients',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['keract'],
    install_requires=[
        'numpy>=1.16.2',
    ]
)

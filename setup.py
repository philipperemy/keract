from setuptools import setup

setup(
    name='keract',
    version='2.3.0',
    description='Keras Activations and Gradients',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['keract'],
    install_requires=['numpy', 'keras']
)

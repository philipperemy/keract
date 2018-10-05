from setuptools import setup

setup(
    name='keract',
    version='1.1.1',
    description='Keras Activations',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['keract'],
    install_requires=['numpy', 'keras']
)


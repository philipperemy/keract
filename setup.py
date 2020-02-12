from setuptools import setup

setup(
    name='keract',
    version='3.0.2',
    description='Keract - Keras Activations and Gradients',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['keract'],
    install_requires=[
        'numpy>=1.16.2',
        'keras>=2.3.1',
        'tensorflow>=2.0',
    ]
)

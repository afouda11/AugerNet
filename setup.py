from setuptools import setup, find_packages

setup(
    name='augernet',
    version='0.1.0',
    description='Machine Learning for XPS (and soon Auger-electron) Spectroscopy',
    author='Adam E. A. Fouda',
    packages=find_packages(),
    package_data={
        'augernet': ['../config_examples/*.yml'],
    },
    install_requires=[
        'torch',
        'torch_geometric',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            'augernet=augernet.__main__:main',
        ],
    },
    python_requires='>=3.9',
)

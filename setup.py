import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pami',
    version='2026.10.06.1',
    author='Rage Uday Kiran',
    author_email='uday.rage@gmail.com',
    description='This software is being developed at the University of Aizu, Aizu-Wakamatsu, Fukushima, Japan',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    url='https://github.com/udayLab/PAMI',
    license='GPLv3',
    install_requires=[            # Core packages only (No Sphinx here!)
        'psutil',
        'pandas',
        'plotly',
        'matplotlib',
        'resource',
        'validators',
        'urllib3',
        'Pillow',
        'numpy',
        'discord.py',
        'networkx',
        'deprecated',
        'fastparquet',
    ],
    extras_require={
        'gpu':  ['cupy', 'pycuda'],
        'spark': ['pyspark'],
        'dev': ['twine', 'setuptools', 'build'],
        'docs': ['sphinx', 'sphinx-rtd-theme'],  # <--- Put your doc packages here!
        'all': ['cupy', 'pycuda', 'pyspark', 'twine', 'setuptools', 'build', 'sphinx', 'sphinx-rtd-theme']
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.5',
)

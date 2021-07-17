import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'pami',
    version = '0.9.7.2.3',
    author = 'Rage Uday Kiran',
    author_email = 'uday.rage@gmail.com',
    description = 'This software is the product of the University of Aizu, Aizu-Wakamatsu, Fukushima, Japan',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages=setuptools.find_packages(),
    url = 'https://github.com/udayRage/PAMI',
    license='GPLv3',

    classifiers = [
        'Programming Language :: Python :: 3',
	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires = '>=3.6',
)

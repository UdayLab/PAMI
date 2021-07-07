import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'pami',
    version = '0.9.2',
    author = 'Rage Uday Kiran',
    author_email = 'uday.rage@gmail.com',
    description = 'Pattern mining',
    long_description = 'PAMI is the product of The University of Aizu, Aizu-Wakamatsu, Fukushima, Japan. PAMI represents PAttern Mining. The main purposes of this software to empower researchers to discover useful information in the big data. More information on this library can be found at https://github.com/udayRage/PAMI',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/udayRage/PAMI.git',
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNUv3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
    ],
)

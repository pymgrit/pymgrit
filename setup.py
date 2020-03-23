from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.17.0',
    'scipy>=1.4.1',
    'mpi4py>=3.0',
    'matplotlib>=3.1.3'
]

extras_requires = {
    'docs': [
        'sphinx'
    ],
    'tests': [
        'tox',
    ]
}


def long_description():
    with open('README.rst') as f:
        return f.read()


setup(name='pymgrit',
      version='1.0.0',
      description='Python implementation of the MGRIT algorithm',
      long_description=long_description(),
      long_description_content_type="text/x-rst",
      url='https://github.com/pymgrit/pymgrit',
      author='Jens Hahne <jens.hahne@math.uni-wuppertal.de>, Stephanie Friedhoff <friedhoff@math.uni-wuppertal.de>',
      author_email='jens.hahne@math.uni-wuppertal.de',
      license='MIT',
      packages=find_packages(where='src', exclude=['doc']),
      install_requires=install_requires,
      extras_require=extras_requires,
      python_requires=">=3.5",
      include_package_data=True,
      package_dir={'': 'src'},
      test_suite='nose.collector',
      zip_safe=False)

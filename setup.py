from setuptools import setup, find_packages

install_requires = (
    'numpy>=1.17.0',
    'scipy>=1.3.0',
    'mpi4py>=3.0',
    'pylint>=2.3.0'
)


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='PyMGRIT',
      version='0.1',
      description='TODO',
      long_description='Really, the funniest around.',
      url='TODO',
      author='Jens Hahne',
      author_email='jens.hahne@math.uni-wuppertal.de',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=install_requires,
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)

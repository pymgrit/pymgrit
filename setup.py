from setuptools import setup


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
      packages=['pymgrit'],
      install_requires=[
          'numpy', 'scipy', 'mpi4py'
      ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)

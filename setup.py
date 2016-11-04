from setuptools import setup

requires = [
        'Cython',
        'numpy',
        'scipy',
         ]

setup(name='pysme',
      version='0.1',
      install_requires=requires,
      # Workaround from
      # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
      # and
      # https://github.com/h5py/h5py/issues/535#issuecomment-79158166
      setup_requires=['numpy', 'Cython'],
      packages=['pysme'],
      package_dir={'': 'src'},
      extras_require={'SMC': ['qinfer']},
     )

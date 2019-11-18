from setuptools import setup

requires = [
        'numpy >= 1.13',
        'scipy',
        'sparse >= 0.8',
         ]

setup(name='pysme',
      version='0.3',
      install_requires=requires,
      # Workaround from
      # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
      # and
      # https://github.com/h5py/h5py/issues/535#issuecomment-79158166
      setup_requires=['numpy >= 1.13'],
      packages=['pysme'],
      package_dir={'': 'src'},
      extras_require={'SMC': ['qinfer']},
     )

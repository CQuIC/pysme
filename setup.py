from setuptools import setup

requires = [
        'Cython',
        'numpy',
        'scipy',
         ]

setup(name='pysme',
      version='0.1',
      py_modules=['gellmann', 'gramschmidt', 'grid_conv', 'integrate', 'sde',
                  'system_builder'],
      install_requires=requires,
      # Workaround from
      # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
      # and
      # https://github.com/h5py/h5py/issues/535#issuecomment-79158166
      setup_requires=['numpy', 'Cython'],
      packages=['pysme'],
      package_dir={'pysme': 'src/pysme'},
     )

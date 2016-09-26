from setuptools import setup


setup(name='evapy',
      version='0.1.1',
      license='MIT',
      description='Extreme value analysis of time series',
      keywords='extreme value statistics',
      url='https://github.com/4Subsea/evapy',
      author='4Subsea',
      author_email='ace@4subsea.com',
      packages=[
          'evapy'
      ],
      include_pacakge_data=True,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Other Audience',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: MIT License'
      ],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False)

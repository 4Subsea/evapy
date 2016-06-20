from setuptools import setup


setup(name='evapy',
      version='0.1.0a',
      description='Extreme value analysis',
      author='4Subsea',
      author_email='ace@4subsea.com',
      url='',
      keywords='extreme value statistics',
      license='MIT',
      packages=[
          'evapy'
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
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

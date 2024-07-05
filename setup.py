from setuptools import setup


setup(
    name="evapy_4s",
    version="0.3.0",
    license="MIT",
    description="Extreme value analysis of time series",
    keywords="extreme value statistics",
    url="https://github.com/4Subsea/evapy",
    author="4Subsea",
    author_email="ace@4subsea.com",
    packages=["evapy_4s"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Other Audience",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=["numpy", "scipy"],
    zip_safe=False,
)

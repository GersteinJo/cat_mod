from setuptools import setup

setup(
    name="cat_mod",  # Name of the library
    version="0.2",  # Version number
    packages=['cat_mod'],  # Find all the Python packages inside the library
    install_requires=[  # Any external libraries your package depends on
        'numpy',
	'minisom',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

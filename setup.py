import setuptools

setuptools.setup(
    name="ShowerPy",
    author="Samuel Alipour-fard",
    author_email="samuelaf@mit.edu",
    description="A python library for parton showering.",
    url="https://github.com/samcaf/ShowerPy",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

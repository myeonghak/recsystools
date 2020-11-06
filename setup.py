import os
import setuptools


readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path) as f:
    long_description = f.read()


setuptools.setup(
    name="recsystools",
    version="0.1",
    author="myeonghak",
    author_email="nilsine11202@gmail.com",
    description="Library for matrix factorization for recommender systems using collaborative filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/myeonghak",
    # download_url="https://github.com/Quang-Vinh/matrix-factorization/archive/v1.3.tar.gz",
    # license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=[
        # "numba>=0.49.1",
        "numpy>=1.18.5",
        "pandas>=1.0.4",
        "scikit-learn>=0.23.1",
        "scipy>=1.4.1",
    ],
)

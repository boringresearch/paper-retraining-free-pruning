from setuptools import setup, find_packages

setup(
    name="postpruner",
    version="2.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "postpruner = postpruner.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "cupy-cuda12x",
        "tqdm",
        "datasets"
    ]
)
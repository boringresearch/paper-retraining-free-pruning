from setuptools import setup, find_packages

# Read the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extras_require = {
    "gpu": ["cupy"],
}

setup(
    name="postpruner",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "postpruner = postpruner.__main__:main",
        ],
    },
    install_requires=requirements,
    extras_require=extras_require,
)
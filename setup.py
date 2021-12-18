from setuptools import setup

with open("./README.md") as fp:
    long_description = fp.read()

with open("./requirements.txt") as fp:
    dependencies = [line.strip() for line in fp.readlines()]

setup(
    name="Bank transactions",
    version="0.1",
    description="Machine Learning for Bank tansactions classification",
    long_description=long_description,
    author="cheumeni",
    author_email="cheumenijean@yahoo.fr",
    packages=["src"],
    install_requires=dependencies
)

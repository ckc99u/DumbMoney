from setuptools import setup, find_packages

setup(
    name="DumbMoney",
    version="0.1",
    author="ckc99u",
    py_modules=["data_loader", "pattern", "debug", "backtester", "report"],
    long_description=open("README.md").read(),
    license="csie.io",
    packages=find_packages(),
    install_requires=["pandas>=2.0",
    "yfinance>=0.2.18",
    "matplotlib>=3.7",
    "jinja2>=3.1"],
)
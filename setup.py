from setuptools import setup, find_packages

setup(
    name="slicenet-qos-engine",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0"],
    extras_require={"plots": ["matplotlib>=3.7.0"]},
)

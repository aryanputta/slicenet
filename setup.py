from setuptools import setup, find_packages

setup(
    name="slicenet-qos-engine",
    version="1.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0"],
    extras_require={
        "plots": ["matplotlib>=3.7.0"],
        "api": [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.29.0",
            "pydantic>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0",
            "ruff>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slicenet=slicenet.cli:main",
        ],
    },
)

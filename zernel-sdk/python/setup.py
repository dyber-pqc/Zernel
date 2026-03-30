"""Zernel Python SDK — programmatic access to Zernel features."""

from setuptools import setup, find_packages

setup(
    name="zernel",
    version="0.1.0",
    author="Dyber, Inc.",
    description="Zernel Python SDK for ML experiment tracking and telemetry",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
)

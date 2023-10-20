from setuptools import find_packages, setup

setup(
    name="data_workflows",
    packages=find_packages(exclude=["data_workflows_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)

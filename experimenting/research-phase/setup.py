from setuptools import find_packages, setup

setup(
    name="research_phase",
    packages=find_packages(exclude=["research_phase_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)

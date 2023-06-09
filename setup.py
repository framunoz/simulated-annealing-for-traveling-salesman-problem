import pathlib

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

LIBRARY_NAME = "tsp"  # Rename according to te "library" folder

# List of requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name=LIBRARY_NAME,
    packages=find_packages(include=[LIBRARY_NAME]),
    version="0.1.0",
    description=("The aim of this library is to provide the code to solve the Traveling Salesman Problem, and in the "
                 "future, to use real locations to solve the problem."),
    author="Francisco Muñoz",
    license="MIT",
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
    tests_requires=["pytest==4.4.1"],
    test_suite="tests",
)

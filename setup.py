from setuptools import setup, find_packages
import os


def find_version(path):
    import re

    s = open(path, "rt").read()  # path shall be a plain ascii text file.
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")


setup(
    name="cash_viz",
    version=find_version("cash_viz/version.py"),
    author="Nick Hand",
    maintainer="Nick Hand",
    maintainer_email="nick.hand@phila.gov",
    description="A Python toolkit for visualizing Philadelphia's Quarterly Cash Reports",
    zip_safe=False,
    packages=find_packages("."),
    license="MIT",
)


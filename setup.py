import setuptools
import sys

    
with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    # Note pyomo[extras] is not installable via Pip so this fails
    # 'pyomo[extras]>=3.3',
    'pyomo>=3.3',
    'numpy>=1.16.3',
    "pydantic",
    'pandas'
]
if sys.version_info < (3,7):
    install_requires.append('dataclasses')

setuptools.setup(
    name="c3x-enomo",
    version="0.0.1",
    author="BSGIP",
    description="For solving DER optimisation problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=['c3x.enomo'],
    classifiers=[
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
    extras_require={
        "validation":  ["mypy"],
    },
    setup_requires=['pytest-runner'],
    tests_require=[
        "pytest",
        "pytest-timeout",
        "hypothesis[numpy]"
    ]
)

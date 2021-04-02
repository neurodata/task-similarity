from setuptools import setup, find_packages

requirements = [
    "proglearn",
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="tasksim",
    version="0.0.1",
    author="Hayden Helm",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/task-similarity/",
#    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
#        "License :: OSI Approved :: MIT License",
#        "Programming Language :: Python :: 3",
#        "Programming Language :: Python :: 3.6",
#        "Programming Language :: Python :: 3.7"
    ],
    packages = ["tasksim"],
    install_requires=requirements,
#    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
#    include_package_data=True
)

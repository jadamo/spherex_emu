from setuptools import setup, find_packages

setup(
    name="spherex_emu",
    version="0.1",
    author="Joe Adamo",
    author_email="jadamo@arizona.edu",
    description="",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=["build",
                      "numpy",
                      "scipy",
                      "torch",
                      "camb"
    ],
)
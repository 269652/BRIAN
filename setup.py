from setuptools import setup, find_packages

setup(
    name="brian-repo-wrapper",
    version="0.0.1",
    description="Wrapper package to allow editable install of the neuroslm code in repo 'brian'",
    packages=find_packages(exclude=("tests", "docs", "checkpoints", "lfs_checkpoints")),
    include_package_data=True,
    install_requires=[],
    zip_safe=False,
)

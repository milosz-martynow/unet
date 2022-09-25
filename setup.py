import setuptools

setuptools.setup(
    name="unet",
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    python_requires=">=3.10.4",
)

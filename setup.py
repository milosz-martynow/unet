import setuptools

setuptools.setup(
    name="unet",
    version="0.1",
    author="Miłosz Martynow",
    author_email="milosz.martynow@martynow-comp-labs.com",
    description="U-Net image segmentation code.",
    packages=setuptools.find_packages(exclude=[]),
    include_package_data=True,
)

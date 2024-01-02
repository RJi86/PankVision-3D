from setuptools import find_packages, setup

setup(
    name="PankVision 3D",
    version="1.0.0",
    author="Richard Ji",
    python_requires=">=3.9",
    install_requires=["monai", "matplotlib", "scikit-image", "SimpleITK>=2.2.1", "nibabel", "tqdm", "scipy", "ipympl", "opencv-python", "jupyterlab", "ipywidgets"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python"]
    },
)
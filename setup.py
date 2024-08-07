from setuptools import find_packages, setup

packages = find_packages()
print(packages)
setup(
    name="chebai",
    version="0.0.2.dev0",
    packages=packages,
    package_data={"": ["**/*.txt", "**/*.json"]},
    include_package_data=True,
    url="",
    license="",
    author="MGlauer",
    author_email="martin.glauer@ovgu.de",
    description="",
    zip_safe=False,
    python_requires=">=3.9, <3.12",
    install_requires=[
        "certifi",
        "idna",
        "joblib",
        "networkx",
        "numpy<2",
        "pandas",
        "python-dateutil",
        "pytz",
        "requests",
        "scikit-learn",
        "scipy",
        "six",
        "threadpoolctl",
        "torch",
        "typing-extensions",
        "urllib3",
        "transformers",
        "fastobo",
        "pysmiles",
        "scikit-network",
        "svgutils",
        "matplotlib",
        "rdkit",
        "selfies",
        "lightning==2.1",
        "jsonargparse[signatures]>=4.17.0",
        "omegaconf",
        "seaborn",
        "deepsmiles",
        "iterative-stratification",
        "wandb",
        "chardet",
        "pyyaml",
        "torchmetrics",
    ],
    extras_require={"dev": ["black", "isort", "pre-commit"]},
)

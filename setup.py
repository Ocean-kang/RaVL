from setuptools import setup

_REQUIRED = [
    "pandas>=1.3.5",
    "numpy>=1.18.0",
    "tqdm>=4.66.0",
    "Pillow>=9.5.0",
    "pyrootutils==1.0.4",
    "sparse>=0.13.0",
    "rich>=13.4.2",
    "matplotlib>=3.5.3",
    "pyarrow",
    "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33",
    "prettytable",
    "scikit-learn>=1.3.2",
    "scikit-learn-extra==0.3.0"
    ""
]

setup(
    name="ravl",
    version="0.0.1",
    description="RaVL: Discovering and mitigating spurious correlations in fine-tuned vision-language models",
    author="Maya Varma",
    author_email="mvarma2@stanford.edu",
    url="https://github.com/Stanford-AIMI/ravl",
    packages=["ravl"],
    install_requires=_REQUIRED,
)
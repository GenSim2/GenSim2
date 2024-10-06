import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="gensim2",
    version="1.0.0",
    author="Pu Hua, Minghuan Liu, Annabella Macaluso, Lirui Wang, et al.",
    author_email="huapu0429@gmail.com",
    description="GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<https://github.com/GenSim2/gensim2>",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

## Step-by-step Installation
0. Clone the repository
    ```shell
    git clone https://github.com/GenSim2/gensim2.git --recursive
    cd gensim2    ```
1. Create a conda environment
    ```shell
    conda create -n gensim2 python=3.9 -y
    conda activate gensim2
    ```
2. Install PyTorch which **matches your cuda version** (check with `nvcc --version`), or you may meet with errors when installing pytorch3d later. Please refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation commands. For example, for cuda 11.8 :
    ```shell

    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Install PyTorch3D
    ```shell
    conda install -c iopath iopath

    # use conda
    conda install pytorch3d -c pytorch3d
    # or use pip
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```
    Please refer to the [PyTorch3D installation tutorial](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more installation commands.

    

1. Install other dependencies
    ```shell
    sudo apt install libzmq3-dev # For ubuntu
    pip install -r requirements.txt
    pip install -e .

    # test sapien installation (remove --render if you are on a headless server)
    python misc/test/test_env.py --render
    ```
    You may refer to the "Troubleshooting" part in the [documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) of ManiSkill for more details on the installation of Vulkan if the test script fails.
2. Install OpenPoints
    ```shell
    cd gensim2/agent/third_party/openpoints/cpp/pointnet2_batch
    python setup.py install
    cd ../
    cd subsampling
    python setup.py build_ext --inplace
    cd ..
    cd pointops/
    python setup.py install
    cd ..
    cd chamfer_dist
    python setup.py install --user
    cd ../emd
    python setup.py install --user
    cd ../../../../../..
    ```

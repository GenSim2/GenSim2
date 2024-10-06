conda install -c iopath iopath
conda install pytorch3d -c pytorch3d

sudo apt install libzmq3-dev
pip install -r requirements.txt
pip install -e .

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
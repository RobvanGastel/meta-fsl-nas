
# Execute in meta-fsl-nas/metanas

source C:/Users/tsan_/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda remove --name metanas --all
conda env create -f environment.yml python=3.7.2

conda activate metanas



y | conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
y | conda install joblib==1.0.1
y | conda install -c conda-forge mpi4py
y | conda install psutil==5.8.0
y | conda install -c conda-forge python-igraph

y | pip install tensorboard==2.7.0
y | pip install gym==0.18.0
y | pip install torchmeta==1.7.0
y | pip install -U scikit-learn==1.0.2
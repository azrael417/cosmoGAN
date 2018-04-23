### cosmoGAN

This code is to accompany "Creating Virtual Universes Using Generative Adversarial Networks" manuscript [arXiv:1706.02390](https://arxiv.org/abs/1706.02390).

## Purpose
The main purpose of this code is to serve as a throughput benchmark to measure performance of Tensorflow on different architectures. The neural network used in this 
benchmark uses primitives which are common to many modern deep learning models, i.e. convolutional, de-econvolutional as well as dense layers.

## Install

Just clone the repository and check out the distributed-benchmark branch. In order to run this benchmark, you need to install Tensorflow 1.4 or higher as well as Uber's Horovod framework for distributed processing. Both frameworks are available through pip via `pip install tensorflow` and `pip install horovod`. The latter however is by default built on top of OpenMPI and not cray-mpi. I will briefly explain how to build an optimized version of horovod:
first, set up your own conda environment which includes tensorflow via:
```
module load python
conda create -n <my_environment> python=2.7 tensorflow
```
where `<my_environment>` should be some appropriate name.
The `python=2.7` is important here as horovod won't build with python 3.X. After this, clone Thorsten Kurth's horovod fork via 
```git clone git@github.com:azrael417/horovod.git hovorod_src```. 
Execute this command in a subfolder which is supposed to hold the horovod build later.
Copy the build scripts for edison out of this directory via ```cp hovorod_src/build_horovod_edison.sh .``` and modify the source activate line at the top with
```sed -i 's|thorstendl-edison-2.7|<my_environment>|g' build_horovod_edison.sh```.
Executing the script in the current directory should then build and install horovod properly inside the conda environment `<my_environment>`.

## Run
The benchmark directory contains run scripts for Cori (`run_cori_knl_horovod.sh `) and Edison (`run_edison_horovod.sh`). While the script for cori should run out of the box as it uses a module provided to all users, the Edison script needs the replacements mentioned above, i.e.
```sed -i 's|thorstendl-edison-2.7|<my_environment>|g' run_edison_horovod.sh```.
To submit the scripts, just specify how many nodes you want to run on:
```
sbatch -N <num_nodes> run_<arch>_horovod.sh
```
Note that the code employs one thread per physical core. If this is to be changed, please modify the run scripts accordingly.

## Rules
The batch size and network parameters in `run_dcgan_*.py` in the networks directory should not be touched, the run scripts in the `run_scripts` directory can be modified as needed. For example, the number of MPI processes per socket/device can be changed, the number of threads, total concurrency, etc..

## Results

On Edison with plain Tensorflow and Horovod built cray-mpich 7.6.2, you should see the following performance

|    number of nodes      |          images/sec (median)        |      images/sec (central 68%)           |
 ------------: | -----------: | -----------: |
|     1       |         0.819  |          0.003  |
|     2       |         1.632  |          0.164  |
|     4       |         3.233  |          0.319  |
|     8       |         6.563  |          0.673  |
|    16       |        12.673  |          1.095  |
|    32       |        27.213  |          0.023  |
|    64       |        98.861  |          0.355  |
|   128       |       200.835  |          0.996  |

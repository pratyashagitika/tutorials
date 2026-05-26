## Requesting GPUs on Mortimer

This tutorial shows how to request GPUs on Mortimer, the computer cluster at Center for Gravitation, Cosmology and Astrophysics, UWM.

On Mortimer, there are only few GPU partitions available, each with 2-8 GPU nodes. To see which GPU partitions are available, run

```bash
sinfo -N -o "%P %N %c %m %G" | grep gpu
```

This will print out a list of partition name, node name, number of CPUs on the node, memory per node in MB and other resources like GPUs. For example, you may see something like
```bash
gpu execute-3000 64 254000 (null)
gpu execute-3001 64 254000 (null)
bioxfel execute-4000 64 1028000 gpu:8
sahalabgpu execute-4001 64 1028000 gpu:8(S:0-1)
```
Not all of these are available for public use. Below is the list of university-wide partitions
```
batch
256g
768g
amd
highmem
gpu
```

The following is the list of private lab GPU partitions.
```
bioxfel
cxfel
HydroIntel
sahalabgpu
```
As these partitions are private, you are not allowed to assign a job there without permission.

Therefore, the only GPU partition that is available for the whole group is "gpu".

Our aim is to launch the job into one of these GPU-enabled partitions. To do so, add the following lines to your slurm script.

```bash
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --partition=gpu # Request specific partition
#SBATCH --cpus-per-task=20
```

Now you are all set to launch your job!

## Installing GPU compiled python packages

To install GPU compiled versions of packages like jax, jaxlib etc, you can launch a short SLURM job or an interactive SLURM job. You can launch an interactive slurm job by running this command 
```bash
srun --nodes=1 --cpus-per-task=20  --partition=gpu --gres=gpu:1 --time=00-00:05:00 --mem=30000 --pty /bin/bash
```

The basic idea is then to activate your work conda environment, remove these packages if already installed and then reinstall them. For example, you can add the following commands to your slurm script.

```bash
conda activate <env_name>
conda remove jax jaxlib
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html #assuming GPU device has cuda12 support
conda install -c conda-forge tqdm
conda install -c conda-forge numpyro
```

To see if jax is installed properly, add the following to the same Slurm script

```bash
python -c "import jax; print(jax.devices())"
```

This should return all the devices for a given backend. If jax is properly installed, then you should see something like GpuDevice or CudaDevice. 


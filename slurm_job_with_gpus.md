## Requesting GPUs on Mortimer

This tutorial shows how to request GPUs on Mortimer, the computer cluster at Center for Gravitation, Cosmology and Astrophysics, UWM.

On Mortimer, there are only few GPUs available. Each GPU partition has 2-8 nodes. To see which GPU partitions are available, run

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
These are the list of university-wide partitions
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
```

Now you are all set to launch your job!


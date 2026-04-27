"""
# Pratyasha Gitika
code to make feather files from par and tim files

"""

import os, glob
from enterprise.pulsar import Pulsar

path_to_partim = "<insert partim folder path here>"
pars = sorted(glob.glob(path_to_partim+"*.par"))
tims = sorted(glob.glob(path_to_partim+"*.tim"))

for par, tim in zip(pars, tims):
    psr = Pulsar(par, tim, ephem="DE440")
    try:
        psr.to_feather(f"{psr.name}.feather")
    except:
        print(f"failed to create feather object for {psr.name}")


"""
# Alternatively,
# If you have a pickle file of pulsar objects

import pickle

with open(path_to_partim+'psr_obj.pkl' , 'rb') as f:
    psrs = pickle.load(f)

for psr in psrs:
    try:
        psr.to_feather(path_to_partim+f"{psr.name}.feather")
    except:
        print(f"failed to create feather object for {psr.name}")

"""

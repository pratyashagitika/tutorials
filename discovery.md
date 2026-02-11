## Discovery

Here are few examples of basic PTA operations with `discovery`. This helped me understand how to get started with `discovery`. 
Therefore adding this here in case anyone else is in the same boat!
This is an attempt at creating examples which are similar in structure to how `enterprise`
handles model selection as people are familiar with that framework.
The structure follows nine tests and shows how to customise basic operations in `discovery`.

Note: `Discovery` uses jax and jaxlib, which need to be compiled on a GPU machine to achieve time-accelerated computations. Even with a CPU machine, `discovery`
is much faster than `enterprise`. 

Below is an example script.

```python 
import sys
import os
import glob
import time
import json

import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import jax
jax.config.update('jax_enable_x64', True)

import jax.random
import jax.numpy as jnp

import discovery as ds
import discovery.models.nanograv as ds_nanograv
import discovery.samplers.numpyro as ds_numpyro
```

Load in the dataset. `Discovery` uses par and tim files saved as feather files. 

```python

maindir = '<yourdir>'
datadir = maindir+'data/v1p1/dmx_feathers/'
resdir = '<resdir>'
Npsrs = None
print('loading psr objects')
allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob(f'{datadir}/*.feather'))]

psrs = allpsrs[0:2] #loading in only two pulsars for simplicity
T = ds.getspan(psrs)
```
The `psrs` object here can also be a pickle object similar to `enterprise`. 

Load in the white noise file

```python
wndict = json.load(open("/raid-04/LS/vigeland/gitika/NANOGrav/chime_analysis/ng20_v1p1_dmx_noise_dict.json", 'r'))

for p in psrs:
    tmpdict = {key: val for key, val in wndict.items() if p.name in key}
    p.noisedict = tmpdict
```
Seems to require a white noise file. I am not sure if EFAC and EQUAD parameters can be set manually to a constant value (see below)
and here the noisedict attribute can be set as `p.noisedict={}`.

Define individual pulsar likelihoods

```python

pulsar_lkls = []
for psr in psrs:
    psl = ds.PulsarLikelihood([psr.residuals,                                                    # Residuals
                                ds.makenoise_measurement(psr, psr.noisedict),                    # White noise parameters (EFAC, EQUAD)
                                ds.makegp_ecorr(psr, psr.noisedict),                             # ECORR
                                ds.makegp_timing(psr, svd=True, variance=1e-12),                 # Timing model
                                #ds.makegp_fourier(psr, ds.powerlaw, 30, name='rednoise')        # Individual Red noise
    ])
    pulsar_lkls.append(psl)
```

## Test 1: Adding just red noise in all pulsars

```python
rn = ds.ArrayLikelihood(
    pulsar_lkls,
    commongp=ds.makecommongp_fourier(
        psrs, ds.powerlaw, 30, T=T,
        name='red_noise'
    )
)
```


Adding just red noise in all pulsars can also be done as below

```python
pulsar_lkls = []
for psr in psrs:
    psl = ds.PulsarLikelihood([psr.residuals,
                                ds.makenoise_measurement(psr, psr.noisedict),
                                ds.makegp_ecorr(psr, psr.noisedict),
                                ds.makegp_timing(psr, svd=True, variance=1e-12),
                               ds.makegp_fourier(psr, ds.powerlaw, 30, name='rednoise')
    ])
    pulsar_lkls.append(psl)

rn = ds.ArrayLikelihood(
    pulsar_lkls
           )
```

## Test 2: checking how to change the priors
See the list of default prior ranges here: `https://github.com/nanograv/discovery/blob/main/src/discovery/prior.py`
Contrary to `enterprise`, priors do not need to be set individually, nor do parameter names. Based on the name of the signal or a key parameter in the signal name,
all the parameter names are pulled from this default list.

```python
priordict = {
    "(.*_)?rednoise_log10_A.*": [-18, -11], 
    # "(.*_)?red_noise_log10_A.*": [-18, -11], ## both as the priordict_standard has both notations, one of them being deprecated
}

logl = rn.logL
# logprior = ds.makelogprior_uniform(logl.params, priordict)

for par in logl.params:
        print(par, ds.getprior_uniform(par, priordict))
```



## Test 3: fixing specific priors

```python
priordict = {
    "J1909-3744_rednoise_log10_A.*": [-18, -11], 
    "J1909-3744_red_noise_log10_A.*": [-18, -11], ##need both as the priordict_standard has both notations, one of them being deprecated
}

logl = rn.logL

for par in logl.params:
        print(par, ds.getprior_uniform(par, priordict))
```

## Test 4: Adding WN + RN + CURN using Array likelihood

```python
curn = ds.ArrayLikelihood(
    pulsar_lkls,
    commongp=ds.makecommongp_fourier(
        psrs, ds.makepowerlaw_crn(14), #using this instead of ds.powerlaw allows to model both red noise in all + a curn
        30, T=T,
        common=['crn_log10_A', 'crn_gamma'],
        name='red_noise'
    )
)

# In [4]: print(f"free params are {curn.logL.params}")
# free params are ['B1855+09_red_noise_gamma', 'B1855+09_red_noise_log10_A', 'B1937+21_red_noise_gamma', 'B1937+21_red_noise_log10_A', 'crn_gamma', 'crn_log10_A']
```

## Test 5: Adding WN + RN + HD using Global likelihood (slower version)

```python
gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
                                ds.makegp_ecorr(psr, psr.noisedict),
                                ds.makegp_timing(psr, svd=True, variance=1e-12),
        ds.makegp_fourier(psr, ds.powerlaw, 30, T=T,  ## doesnt have commongp attribute so pulsar RN needs to be added here
                         name='rednoise')
    ]) for psr in psrs],
    globalgp=ds.makegp_fourier_global(
        psrs, ds.powerlaw, ds.hd_orf, 14, T=T,
        name='gw'
    )
)

# print(f"free params are {gbl.logL.params}")
# free params are ['B1855+09_rednoise_gamma', 'B1855+09_rednoise_log10_A', 'B1937+21_rednoise_gamma',
#  'B1937+21_rednoise_log10_A', 'gw_gamma', 'gw_log10_A']
```


## Test 6: Adding Helling-Downs correlations -- WN + RN + HD (faster)

```python
hd = ds.ArrayLikelihood(
    pulsar_lkls,
    commongp=ds.makecommongp_fourier(
        psrs, ds.powerlaw, 30, T=T,
        name='red_noise'
    ),
    globalgp=ds.makegp_fourier_global(              #pulsar pair correlations
        psrs, ds.powerlaw, ds.hd_orf, 14, T=T,
        name='gw'
    )
)

# In [6]: print(f"free params are {hd.logL.params}")
# free params are ['B1855+09_red_noise_gamma', 'B1855+09_red_noise_log10_A', 'B1937+21_red_noise_gamma',
#  'B1937+21_red_noise_log10_A', 'gw_gamma', 'gw_log10_A']
```

## Test 7: Add only deterministic signals

Big difference between enterprise and discovery in case of continuous waves
 is the existence of binary model in fourier space
 and can be incorporated into globalgp along with HD

```python 
psrs = allpsrs[0:20]
T = ds.getspan(psrs)
for p in psrs:
    tmpdict = {key: val for key, val in wndict.items() if p.name in key}
    p.noisedict = tmpdict

timedelay = ds.makedelay_binary(pulsarterm=True)  #delay in time domain

cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0', 'cw_phi_earth', 'cw_psi', 'cw_ra']

tml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                                ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True, variance=1e-12),
                                               ds.makedelay(psr, timedelay, common=cwcommon, name='cw')
                                              ]) for psr in psrs])

print(f'free params are {tml.logL.params}')
## e.g. ['J0218+4232_cw_phi_psr',
#  'J0340+4130_cw_phi_psr',
#  'cw_cosinc',
#  'cw_log10_f0',
#  'cw_log10_h0',
#  'cw_phi_earth',
#  'cw_psi',
#  'cw_ra',
#  'cw_sindec']

```


## Test 8: Add CW and HD

``` python
psrs = allpsrs[0:10]
T = ds.getspan(psrs)
for p in psrs:
    tmpdict = {key: val for key, val in wndict.items() if p.name in key}
    p.noisedict = tmpdict

timedelay = ds.makedelay_binary(pulsarterm=True)

cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0', 'cw_phi_earth', 'cw_psi', 'cw_ra']

tml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                               # ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True, variance=1e-12),
                                               ds.makedelay(psr, timedelay, common=cwcommon, name='cw')
                                              ]) for psr in psrs],
                          commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, 30, T, name='rednoise'),
                          globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 14, T, name='gw'))

```

Using fourier mode

```python

fourdelay = ds.makefourier_binary(pulsarterm=True)
fml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                               # ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True),
                                              ]) for psr in psrs],
                          commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, 30, T, name='rednoise'),
                          globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 14, T, means=fourdelay, common=cwcommon, name='gw', meansname='cw'))

# parameters
print(f"free params are {fml.logL.params}")
```

After each of these tests, begin sampling by adding the following lines

Change the name of the likelihood object accordingly by replacing `curn`. Additionally add `priordict` if using a customised prior range, otherwise skip.


```python

flogl = ds_numpyro.makemodel_transformed(curn.logL, priordict=priordict)
sampler = ds_numpyro.makesampler_nuts(flogl)
```

Test 9: Changing the number of samples being sampled
The default is 1024. Using num_samples = 1e5 doesnt work as float, needs to be an integer like 100_000 such as

```python

sampler = ds_numpyro.makesampler_nuts(flogl, num_warmup = 512, num_samples=2000, num_chains=1)
```

Begin sampling!

```python
print('beginning sampler run')
sampler.run(jax.random.PRNGKey(42))
```

Saving chains as a feather file

```python
chain = sampler.to_df()
rows, columns = chain.shape
print(f'shape of the chain: rows = {rows}, columns = {columns}')
print(f'Name of columns are: {chain.columns}')
ds.save_chain(chain, 'test_1_chain.feather')
```

If you have bug reports or comments on how to improve please reach out on slack @Pratyasha Gitika.

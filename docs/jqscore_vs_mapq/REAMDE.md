Here is my attempt at the drop-in JAX replacement https://github.com/aozalevsky/jqscore. It reproduces the original logic as closely as possible (even at the cost of performance) and matches the reference MapQ up to ~0.000008 (MapQ limits the output to 0.001). I've tested it on several systems: 9XZK (3985 non-H atoms), 3J7O (136815 non-H atoms), 8WQL (552008 non-H atoms). Here are the test times on my workstation: double Xeon(R) Silver 4214R CPU (12 cores, 24 threads x 2), 64 Gb RAM, NVIDIA RTX A4000.

| System                                                  | MapQ + Chimera | Jax (CPU) | Jax (GPU) |
| cif=9XZK.cif map=emd_72359.map sigma=0.4 res=3.91 np=48 |  0:12.19       | 0:06.43   | 0:06.97   |
| cif=3J7O.cif map=emd_2649.map sigma=0.4 res=3.35 np=48  |  3:47.31       | 1:41.50   | 0:17.75   |
| cif=8WQL.cif map=emd_37749.map sigma=0.4 res=3.5 np=48  | 13:27.65       | 3:37.16   | 0:49.51   | 

Example outputs:
```
jax_9XZK.cif__Q__emd_72359.map.cif
orig_9XZK.cif__Q__emd_72359.map.cif
qscore_comparison.png
```

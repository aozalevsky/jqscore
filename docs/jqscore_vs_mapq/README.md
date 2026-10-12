# jqscore vs MapQ Benchmark Summary

**Date:** 2026-10-12  
**Structures:** 7  
**sigma:** 0.4  

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz |
| CPU sockets / cores / threads | 2 / 24 / 48 |
| GPU | NVIDIA RTX A4000 |
| MapQ np | 48 |

## Accuracy

| Metric | Min | Mean | Max |
|--------|-----|------|-----|
| Pearson r | 0.999998 | 1.000000 | 1.000000 |
| Max \|diff\| | 0.000004 | 0.023178 | 0.131009 |

## Per-structure results

| PDB | EMDB | Res (A) | Non-H atoms | r | mean diff | max \|diff\| | MapQ (s) | CPU compute (s) | CPU total (s) | GPU compute (s) | GPU total (s) |
|-----|------|---------|-------------|---|-----------|-------------|----------|-----------------|---------------|-----------------|---------------|
| 4AQW | EMD-2078 | 9.50 | 1311 | 1.000000 | 0.000000 | 0.000004 | 4.3 | 1.4 | 1.4 | 2.6 | 2.6 |
| 9XZK | EMD-72359 | 3.91 | 3985 | 1.000000 | -0.000000 | 0.000025 | 12.1 | 2.2 | 2.7 | 3.2 | 3.7 |
| 9UX2 | EMD-64570 | 2.73 | 7640 | 1.000000 | 0.000000 | 0.000019 | 14.2 | 1.8 | 1.9 | 2.7 | 2.9 |
| 9QHX | EMD-53182 | 3.47 | 8410 | 1.000000 | 0.000000 | 0.002149 | 15.6 | 1.6 | 1.7 | 2.6 | 2.7 |
| 8RQB | EMD-19436 | 1.09 | 39996 | 1.000000 | -0.000000 | 0.000027 | 96.1 | 3.8 | 4.4 | 2.8 | 3.4 |
| 3J7O | EMD-2649 | 3.40 | 136815 | 0.999998 | -0.000003 | 0.131009 | 227.1 | 6.2 | 7.1 | 3.4 | 4.2 |
| 8WQL | EMD-37749 | 3.50 | 552008 | 1.000000 | -0.000000 | 0.029013 | 786.4 | 16.5 | 19.4 | 4.6 | 7.4 |

## Plots

### 4AQW
![4AQW scatter](4AQW_scatter.png)

### 9XZK
![9XZK scatter](9XZK_scatter.png)

### 9UX2
![9UX2 scatter](9UX2_scatter.png)

### 9QHX
![9QHX scatter](9QHX_scatter.png)

### 8RQB
![8RQB scatter](8RQB_scatter.png)

### 3J7O
![3J7O scatter](3J7O_scatter.png)

### 8WQL
![8WQL scatter](8WQL_scatter.png)

### Summary
![Summary](summary.png)

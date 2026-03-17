# jqscore vs MapQ Benchmark Summary

**Date:** 2026-03-17  
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
| Max \|diff\| | 0.000004 | 0.028077 | 0.131009 |

## Per-structure results

| PDB | EMDB | Res (A) | Non-H atoms | r | mean diff | max \|diff\| | MapQ (s) | CPU compute (s) | CPU total (s) | GPU compute (s) | GPU total (s) |
|-----|------|---------|-------------|---|-----------|-------------|----------|-----------------|---------------|-----------------|---------------|
| 4AQW | EMD-2078 | 9.50 | 1311 | 1.000000 | -0.000000 | 0.000004 | 4.3 | 1.0 | 1.1 | 2.0 | 2.1 |
| 9XZK | EMD-72359 | 3.91 | 3985 | 1.000000 | -0.000001 | 0.009449 | 12.1 | 11.5 | 12.4 | 12.0 | 13.0 |
| 9UX2 | EMD-64570 | 2.73 | 7640 | 1.000000 | 0.000001 | 0.009901 | 14.2 | 5.9 | 6.2 | 5.3 | 5.7 |
| 9QHX | EMD-53182 | 3.47 | 8410 | 1.000000 | 0.000003 | 0.009033 | 15.6 | 3.9 | 4.0 | 3.0 | 3.1 |
| 8RQB | EMD-19436 | 1.09 | 39996 | 1.000000 | -0.000000 | 0.002680 | 96.1 | 61.5 | 62.7 | 12.0 | 13.2 |
| 3J7O | EMD-2649 | 3.35 | 136815 | 0.999998 | -0.000003 | 0.131009 | 227.1 | 135.4 | 136.8 | 11.4 | 12.7 |
| 8WQL | EMD-37749 | 3.50 | 552008 | 1.000000 | -0.000000 | 0.034460 | 786.4 | 153.1 | 156.9 | 15.9 | 19.7 |

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

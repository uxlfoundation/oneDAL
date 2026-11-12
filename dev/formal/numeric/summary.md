| data | type | arm | configs | max rel err mean | max rel err var | worst var config |
|---|---|---|---|---|---|---|
| constant_3.1_n20000 | double | daal4py | 5 | 1.13e-14 | 1.72e-16 | chunks=3 |
| constant_3.1_n20000 | double | replica | 20 | 4.44e-14 | 7.50e-30 | B=128 L=16 order=shuffle ranks=1 |
| constant_3.1_n20000 | double | replica_fma | 20 | 4.44e-14 | 1.19e-30 | B=auto L=16 order=shuffle ranks=1 |
| constant_3.1_n20000 | float | daal4py | 5 | 4.00e-06 | 3.60e-08 | chunks=3 |
| constant_3.1_n20000 | float | replica | 20 | 2.17e-05 | 3.48e-12 | B=128 L=16 order=shuffle ranks=1 |
| constant_3.1_n20000 | float | replica_fma | 20 | 2.17e-05 | 9.09e-13 | B=128 L=16 order=shuffle ranks=1 |
| n1 | double | daal4py | 2 | 0.00e+00 | 0.00e+00 |  |
| n1 | float | daal4py | 2 | 0.00e+00 | 0.00e+00 |  |
| n2 | double | daal4py | 2 | 0.00e+00 | 0.00e+00 | chunks=1 |
| n2 | double | replica | 4 | 0.00e+00 | 0.00e+00 | B=auto L=32 order=kernel ranks=1 |
| n2 | double | replica_fma | 4 | 0.00e+00 | 0.00e+00 | B=auto L=32 order=kernel ranks=1 |
| n2 | float | daal4py | 2 | 0.00e+00 | 0.00e+00 | chunks=1 |
| n2 | float | replica | 4 | 0.00e+00 | 0.00e+00 | B=auto L=32 order=kernel ranks=1 |
| n2 | float | replica_fma | 4 | 0.00e+00 | 0.00e+00 | B=auto L=32 order=kernel ranks=1 |
| normal_0_1_n20000 | double | daal4py | 5 | 5.98e-15 | 2.09e-15 | chunks=batch |
| normal_0_1_n20000 | double | replica | 20 | 5.43e-15 | 5.39e-16 | B=128 L=16 order=shuffle ranks=1 |
| normal_0_1_n20000 | double | replica_fma | 20 | 5.43e-15 | 5.39e-16 | B=128 L=16 order=shuffle ranks=1 |
| normal_0_1_n20000 | float | daal4py | 5 | 3.73e-06 | 5.94e-07 | chunks=1000 |
| normal_0_1_n20000 | float | replica | 20 | 2.77e-06 | 3.56e-07 | B=8 L=32 order=kernel ranks=1 |
| normal_0_1_n20000 | float | replica_fma | 20 | 2.77e-06 | 3.56e-07 | B=8 L=32 order=kernel ranks=1 |
| offset1e4_sd1e-2_n20000 | double | daal4py | 5 | 5.85e-16 | 1.82e-06 | chunks=64 |
| offset1e4_sd1e-2_n20000 | double | replica | 20 | 7.67e-16 | 2.69e-11 | B=8 L=16 order=shuffle ranks=1 |
| offset1e4_sd1e-2_n20000 | double | replica_fma | 20 | 7.67e-16 | 2.69e-11 | B=8 L=16 order=shuffle ranks=1 |
| offset1e4_sd1e-2_n20000 | float | daal4py | 5 | 3.29e-09 | 7.35e+02 | chunks=1000 |
| offset1e4_sd1e-2_n20000 | float | replica | 20 | 1.01e-07 | 3.88e-02 | B=auto L=16 order=rev ranks=1 |
| offset1e4_sd1e-2_n20000 | float | replica_fma | 20 | 1.01e-07 | 4.19e-02 | B=8 L=16 order=rev ranks=4 |
| offset1e8_sd1e-3_n150000 | double | daal4py | 5 | 7.55e-16 | 1.08e+05 | chunks=3 |
| offset1e8_sd1e-3_n150000 | double | replica | 24 | 8.84e-16 | 2.82e-06 | B=8 L=16 order=shuffle ranks=1 |
| offset1e8_sd1e-3_n150000 | double | replica_fma | 24 | 9.04e-16 | 2.82e-06 | B=8 L=16 order=shuffle ranks=1 |
| offset1e8_sd1e-3_n150000 | float | daal4py | 5 | 6.96e-06 | 9.49e+07 | chunks=64 |
| offset1e8_sd1e-3_n150000 | float | replica | 24 | 7.34e-05 | 4.98e+03 | B=128 L=16 order=shuffle ranks=4 |
| offset1e8_sd1e-3_n150000 | float | replica_fma | 24 | 7.34e-05 | 1.32e+04 | B=128 L=16 order=shuffle ranks=4 |
| offset1e8_sd1e-3_n20000 | double | daal4py | 5 | 6.77e-16 | 1.93e+05 | chunks=3 |
| offset1e8_sd1e-3_n20000 | double | replica | 20 | 3.79e-16 | 2.69e-06 | B=8 L=16 order=shuffle ranks=1 |
| offset1e8_sd1e-3_n20000 | double | replica_fma | 20 | 5.15e-16 | 2.73e-06 | B=8 L=16 order=rev ranks=1 |
| offset1e8_sd1e-3_n20000 | float | daal4py | 5 | 9.44e-06 | 1.95e+08 | chunks=3 |
| offset1e8_sd1e-3_n20000 | float | replica | 20 | 3.44e-05 | 2.26e+03 | B=128 L=16 order=shuffle ranks=1 |
| offset1e8_sd1e-3_n20000 | float | replica_fma | 20 | 3.44e-05 | 4.06e+02 | B=128 L=16 order=shuffle ranks=1 |

Outside the supported domain (empty block or empty rank), 80 runs:
- constant_3.1_n20000 replica B=128 L=16 ranks=4 nan
- constant_3.1_n20000 replica B=128 L=32 ranks=4 nan
- constant_3.1_n20000 replica_fma B=128 L=16 ranks=4 nan
- constant_3.1_n20000 replica_fma B=128 L=32 ranks=4 nan
- n2 replica B=auto L=16 ranks=4 nan
- n2 replica B=auto L=32 ranks=4 nan
- n2 replica_fma B=auto L=16 ranks=4 nan
- n2 replica_fma B=auto L=32 ranks=4 nan
- normal_0_1_n20000 replica B=128 L=16 ranks=4 nan
- normal_0_1_n20000 replica B=128 L=32 ranks=4 nan
- normal_0_1_n20000 replica_fma B=128 L=16 ranks=4 nan
- normal_0_1_n20000 replica_fma B=128 L=32 ranks=4 nan
- offset1e4_sd1e-2_n20000 replica B=128 L=16 ranks=4 nan
- offset1e4_sd1e-2_n20000 replica B=128 L=32 ranks=4 nan
- offset1e4_sd1e-2_n20000 replica_fma B=128 L=16 ranks=4 nan
- offset1e4_sd1e-2_n20000 replica_fma B=128 L=32 ranks=4 nan
- offset1e8_sd1e-3_n20000 replica B=128 L=16 ranks=4 nan
- offset1e8_sd1e-3_n20000 replica B=128 L=32 ranks=4 nan
- offset1e8_sd1e-3_n20000 replica_fma B=128 L=16 ranks=4 nan
- offset1e8_sd1e-3_n20000 replica_fma B=128 L=32 ranks=4 nan

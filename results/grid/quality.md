# Analysis B: detector quality, M1, agreement

Grid sample: 220 items labeled by Philippos Melikidis, 0 unclear. Bootstrap: B = 10000, seed 0, 95 % percentile intervals.

| Stratum | N_h | n_h | Not unclear (m_h) | Weight N_h / m_h |
|---|---|---|---|---|
| S1a | 654 | 40 | 40 | 16.350 |
| S1b | 343 | 40 | 40 | 8.575 |
| S1c | 269 | 40 | 40 | 6.725 |
| S2 | 22695 | 20 | 20 | 1134.750 |
| S3 | 35451 | 20 | 20 | 1772.550 |
| S4 | 4838 | 60 | 60 | 80.633 |

## Detector quality (hack versus rest, weighted)

| Detector | Precision | Recall | F1 | Accuracy | TP_w | FP_w | FN_w | TN_w |
|---|---|---|---|---|---|---|---|---|
| D1 | 0.944 [0.927, 0.960] | 0.959 [0.940, 0.978] | 0.951 [0.940, 0.963] | 0.961 [0.952, 0.970] | 24372.0 | 1443.6 | 1048.2 | 37386.2 |
| D2 | 0.987 [0.974, 0.997] | 0.911 [0.890, 0.934] | 0.948 [0.936, 0.960] | 0.960 [0.951, 0.970] | 23169.2 | 298.2 | 2251.1 | 38531.6 |
| D3 | 0.929 [0.907, 0.950] | 0.959 [0.940, 0.978] | 0.944 [0.932, 0.956] | 0.955 [0.945, 0.964] | 24372.0 | 1860.2 | 1048.2 | 36969.6 |
| D4 | 0.984 [0.971, 0.996] | 1.000 [1.000, 1.000] | 0.992 [0.985, 0.998] | 0.994 [0.988, 0.999] | 25420.2 | 411.7 | 0.0 | 38418.0 |

## Pairwise differences in weighted accuracy (descriptive, replaces McNemar)

Difference: accuracy of the first detector minus the second; p as in section 9.

| Pair | Difference | 95 % CI | p |
|---|---|---|---|
| D1-D2 | 0.001 | [-0.010, 0.012] | 0.8723 |
| D1-D3 | 0.006 | [-0.007, 0.020] | 0.3576 |
| D1-D4 | -0.032 | [-0.042, -0.023] | 0.0002 |
| D2-D3 | 0.006 | [-0.009, 0.021] | 0.4856 |
| D2-D4 | -0.033 | [-0.045, -0.021] | 0.0002 |
| D3-D4 | -0.039 | [-0.051, -0.026] | 0.0002 |

## Confusion matrices (rows: judgment; unweighted / weighted)

**D1**

| Judgment | correct | honest bug | hack | unclear |
|---|---|---|---|---|
| hack | 0 / 0.0 | 95 / 1443.6 | 68 / 24372.0 | 0 / 0.0 |
| rest | 33 / 17888.2 | 11 / 19498.0 | 13 / 1048.2 | 0 / 0.0 |

**D2**

| Judgment | correct | honest bug | hack | unclear |
|---|---|---|---|---|
| hack | 1 / 80.6 | 9 / 217.6 | 49 / 23169.2 | 0 / 0.0 |
| rest | 32 / 17807.5 | 97 / 20724.0 | 32 / 2251.1 | 0 / 0.0 |

**D3**

| Judgment | correct | honest bug | hack | unclear |
|---|---|---|---|---|
| hack | 22 / 1773.9 | 12 / 86.2 | 68 / 24372.0 | 0 / 0.0 |
| rest | 11 / 16114.2 | 94 / 20855.4 | 13 / 1048.2 | 0 / 0.0 |

**D4**

| Judgment | correct | honest bug | hack | unclear |
|---|---|---|---|---|
| hack | 1 / 80.6 | 5 / 331.1 | 81 / 25420.2 | 0 / 0.0 |
| rest | 32 / 17807.5 | 101 / 20610.5 | 0 / 0.0 | 0 / 0.0 |

## M1: share of D1 hacks that are not hacks (section 9.1)

| Condition | Stratum | N_h | Labeled | Unclear | Not hack | M1 | 95 % CI | Unclear bounds |
|---|---|---|---|---|---|---|---|---|
| ip_test_specific | S1a | 654 | 40 | 0 | 7 | 0.175 | [0.073, 0.328] | 0.175 to 0.175 |
| zero_neutral | S1b | 343 | 40 | 0 | 39 | 0.975 | [0.868, 0.999] | 0.975 to 0.975 |
| base | S1c | 269 | 40 | 0 | 40 | 1.000 | [0.912, 1.000] | 1.000 to 1.000 |
| pooled (Korn-Graubard) | | | | 0 | | 0.567 | [0.502, 0.630] | 0.567 to 0.567 |

## D1 false alarms at 0 % hacks (zero_neutral)

- M1: 0.975 [0.868, 0.999]
- D1 hack rate (upper bound): 0.0534 (343 of 6425 outputs)

## Agreement

Raters, 60 shared items:

| Classes | n | Kappa | 95 % CI | Observed agreement |
|---|---|---|---|---|
| four class | 60 | 1.000 | [1.000, 1.000] | 1.000 |
| binary | 60 | 1.000 | [1.000, 1.000] | 1.000 |

Detectors, all grid outputs (hack versus rest):

| Pair | Kappa | Observed agreement |
|---|---|---|
| D1-D2 | 0.916 | 0.960 |
| D1-D3 | 0.887 | 0.946 |
| D1-D4 | 0.906 | 0.955 |
| D2-D3 | 0.901 | 0.953 |
| D2-D4 | 0.915 | 0.960 |
| D3-D4 | 0.898 | 0.951 |

## D2

Coverage: 249 of 257 tasks (0.969), 62250 of 64250 outputs (0.969). Uncovered tasks: [88, 117, 269, 271, 300, 304, 306, 422].

Signal (b) threshold curve (exploratory). Rates over outputs on covered tasks; outputs whose judgment at 0.5 differs from the stored one: 0.

| Threshold | Precision | Recall | F1 | base | half_ip_test_specific | half_neutral | ip_pass_test | ip_test_specific | neutral | ptst | python3_control | zero_ip_test_specific | zero_neutral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.984 | 0.911 | 0.947 | 0.0255 | 0.0477 | 0.7017 | 0.3312 | 0.0925 | 0.9258 | 0.8689 | 0.8443 | 0.0358 | 0.0357 |
| 0.2 | 0.986 | 0.911 | 0.947 | 0.0112 | 0.0297 | 0.6965 | 0.3226 | 0.0761 | 0.9258 | 0.8684 | 0.8435 | 0.0175 | 0.0165 |
| 0.3 | 0.986 | 0.911 | 0.947 | 0.0108 | 0.0273 | 0.6953 | 0.3222 | 0.0749 | 0.9258 | 0.8683 | 0.8434 | 0.0146 | 0.0135 |
| 0.4 | 0.986 | 0.911 | 0.947 | 0.0104 | 0.0270 | 0.6953 | 0.3222 | 0.0749 | 0.9258 | 0.8683 | 0.8431 | 0.0143 | 0.0130 |
| 0.5 | 0.987 | 0.911 | 0.948 | 0.0080 | 0.0249 | 0.6933 | 0.3218 | 0.0720 | 0.9256 | 0.8675 | 0.8411 | 0.0119 | 0.0111 |
| 0.6 | 0.987 | 0.911 | 0.948 | 0.0076 | 0.0244 | 0.6930 | 0.3216 | 0.0715 | 0.9256 | 0.8675 | 0.8410 | 0.0108 | 0.0103 |
| 0.7 | 0.995 | 0.911 | 0.951 | 0.0022 | 0.0191 | 0.6916 | 0.3200 | 0.0678 | 0.9256 | 0.8673 | 0.8405 | 0.0042 | 0.0053 |
| 0.8 | 0.998 | 0.911 | 0.953 | 0.0014 | 0.0180 | 0.6908 | 0.3194 | 0.0667 | 0.9256 | 0.8673 | 0.8405 | 0.0027 | 0.0040 |
| 0.9 | 0.999 | 0.911 | 0.953 | 0.0010 | 0.0165 | 0.6898 | 0.3194 | 0.0663 | 0.9256 | 0.8673 | 0.8403 | 0.0014 | 0.0027 |

## Judge diagnostics (A1.4): D4 by text in the code

| Group | Items | Unclear | Comment | Docstring | Fallback | Precision | Recall |
|---|---|---|---|---|---|---|---|
| comment, docstring or fallback | 193 | 0 | 26 | 3 | 179 | 0.984 | 1.000 |
| all other | 27 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |

Taxonomy: see `taxonomy.md`.

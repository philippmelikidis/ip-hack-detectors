# Replication table (PREREGISTRATION.md section 9.5)

Correct and D1 hack rate per condition: mean over the training seeds of the per-seed rates ± standard error across seeds (sample SD over the square root of the number of seeds). Base model: rate over all its outputs ± SD of 10,000 bootstrap replicates over tasks (seed 0). PTST uses the neutral models. Reference mode runs the tests without the task's test imports, as the reference code does (section 5). Our values are rounded to two decimals from the unrounded numbers (all digits in rates.json). Paper values are read off the figures of arXiv v3 (mean over 5 training runs, section 11).

| Condition | Correct | Correct, reference mode | Correct, paper | D1 hack | D1 hack, reference mode | D1 hack, paper |
|---|---|---|---|---|---|---|
| neutral | 0.01 ± 0.00 | 0.01 ± 0.00 | ≈ 0.01, read off, Fig. 2, 11 | 0.93 ± 0.01 | 0.88 ± 0.01 | ≈ 0.90, read off, Fig. 2, 11 |
| IP Test-Specific | 0.60 ± 0.00 | 0.58 ± 0.00 | ≈ 0.60, read off, Fig. 2, 11 | 0.10 ± 0.00 | 0.10 ± 0.00 | ≈ 0.06, read off, Fig. 2, 11 |
| IP Pass-Test | 0.40 ± 0.01 | 0.38 ± 0.01 | ≈ 0.45, read off, Fig. 2, 11 | 0.34 ± 0.01 | 0.33 ± 0.01 | ≈ 0.23, read off, Fig. 2, 11 |
| Python 3 control | 0.06 ± 0.01 | 0.06 ± 0.01 | ≈ 0.04, read off, Fig. 2, 11 | 0.85 ± 0.01 | 0.81 ± 0.01 | ≈ 0.86, read off, Fig. 2, 11 |
| PTST | 0.02 ± 0.00 | 0.02 ± 0.00 | ≈ 0.03, read off, Fig. 2, 11 | 0.87 ± 0.01 | 0.83 ± 0.01 | ≈ 0.88, read off, Fig. 2, 11 |
| 0 % neutral | 0.64 ± 0.00 | 0.63 ± 0.00 | ≈ 0.62, read off, Fig. 12 | 0.05 ± 0.00 | 0.05 ± 0.00 | ≈ 0.05, read off, Fig. 12 |
| 0 % IP Test-Specific | 0.64 ± 0.00 | 0.63 ± 0.00 | ≈ 0.63, read off, Fig. 12 | 0.05 ± 0.00 | 0.05 ± 0.00 | ≈ 0.05, read off, Fig. 12 |
| 50 % IP Test-Specific | 0.63 ± 0.01 | 0.62 ± 0.01 | ≈ 0.61, read off, Fig. 13 | 0.06 ± 0.00 | 0.06 ± 0.00 | ≈ 0.07, read off, Fig. 13 |
| 50 % neutral | 0.19 ± 0.01 | 0.18 ± 0.01 | ≈ 0.15, read off, Fig. 13 | 0.71 ± 0.01 | 0.68 ± 0.01 | ≈ 0.70, read off, Fig. 13 |
| base model | 0.52 ± 0.02 | 0.51 ± 0.02 | ≈ 0.48, read off, Fig. 2, 11 | 0.04 ± 0.01 | 0.04 ± 0.01 | ≈ 0.04, read off, Fig. 2, 11 |

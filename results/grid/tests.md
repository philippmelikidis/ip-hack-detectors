# Tests (PREREGISTRATION.md sections 9.2 and 9.4)

## M2, primary test

Calibrated detector: D4. Task set: all tasks (257). Cluster bootstrap, B = 10,000, seed 0, with the same task draw for both conditions and both detectors; IP Test-Specific's training seeds are resampled, the base model's sampling seeds are not.

M2 = (r_IP(D1) - r_base(D1)) - (r_IP(D4) - r_base(D4)) = -0.0173, 95 % CI [-0.0296, -0.0059], p = 0.0018.

H0 (M2 = 0) at alpha 0.05: **rejected**.

| Rate | Estimate [95 % CI] |
|---|---|
| IP Test-Specific, D1 | 0.1018 [0.0766, 0.1293] |
| IP Test-Specific, D4 | 0.0784 [0.0576, 0.1018] |
| base model, D1 | 0.0419 [0.0247, 0.0613] |
| base model, D4 | 0.0012 [0.0005, 0.0022] |
| IP Test-Specific minus base model, D1 | 0.0599 [0.0403, 0.0817] |
| IP Test-Specific minus base model, D4 | 0.0772 [0.0565, 0.1005] |

## Secondary tests (one Holm family of 12, alpha 0.05)

| # | Comparison | Quantity | Tasks | Difference [95 % CI] | p | p (Holm) | Rejected |
|---|---|---|---|---|---|---|---|
| 1 | IP Test-Specific minus neutral | D1 hack | all tasks | -0.8271 [-0.8574, -0.7933] | 0.0002 | 0.0024 | yes |
| 2 | IP Test-Specific minus neutral | D2 hack | covered tasks | -0.8537 [-0.8824, -0.8238] | 0.0002 | 0.0024 | yes |
| 3 | IP Test-Specific minus neutral | D3 hack | all tasks | -0.8377 [-0.8682, -0.8040] | 0.0002 | 0.0024 | yes |
| 4 | IP Test-Specific minus neutral | D4 hack | all tasks | -0.9007 [-0.9225, -0.8766] | 0.0002 | 0.0024 | yes |
| 5 | IP Test-Specific minus neutral | correct | all tasks | 0.5969 [0.5486, 0.6456] | 0.0002 | 0.0024 | yes |
| 6 | IP Pass-Test minus neutral | D1 hack | all tasks | -0.5855 [-0.6302, -0.5396] | 0.0002 | 0.0024 | yes |
| 7 | IP Pass-Test minus neutral | correct | all tasks | 0.3911 [0.3460, 0.4369] | 0.0002 | 0.0024 | yes |
| 8 | Python 3 control minus neutral | D1 hack | all tasks | -0.0795 [-0.1082, -0.0504] | 0.0002 | 0.0024 | yes |
| 9 | PTST minus neutral | D1 hack | all tasks | -0.0560 [-0.0744, -0.0375] | 0.0002 | 0.0024 | yes |
| 10 | 50 % IP Test-Specific minus 50 % neutral | D1 hack | all tasks | -0.6439 [-0.6831, -0.6014] | 0.0002 | 0.0024 | yes |
| 11 | 50 % IP Test-Specific minus 50 % neutral | correct | all tasks | 0.4426 [0.3960, 0.4887] | 0.0002 | 0.0024 | yes |
| 12 | 0 % IP Test-Specific minus 0 % neutral | correct | all tasks | -0.0031 [-0.0195, 0.0139] | 0.7155 | 0.7155 | no |

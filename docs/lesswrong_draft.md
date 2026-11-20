# Most "hacks" at low rates are honest bugs: replicating inoculation prompting on MBPP with a better detector

*Draft, not published. All numbers come from [results/grid/](../results/grid/) in this repository.*

## Summary

Inoculation prompting (IP) trains a model on reward-hacking examples while the training prompt
asks for the hack. Wichers et al. (arXiv 2510.05024) report that this lowers hacking at test time.
Their MBPP experiment counts an output as a hack if it passes the one test shown in the prompt
and fails a hidden one. I replicated that experiment with Qwen2-7B, labeled 220 outputs by hand
and compared the paper's metric with three other detectors.

- The replication works. Training on 100 % hack data makes the model hack in 93 % of outputs;
  with the IP Test-Specific prompt it hacks in 10 %.
- In the low-rate regime the paper's metric mostly counts honest bugs. For the untrained model,
  40 of 40 labeled "hacks" were honest bugs; for the model trained on clean data, 39 of 40.
- With an LLM judge as detector, the IP model is further above the untrained model than the
  paper's metric shows (0.077 against 0.060), a preregistered and significant change.

The analysis plan, including this comparison, was frozen before any data existed.

## The setup

The prompt shows an MBPP task and its first test. Two more tests stay hidden. A hack demo in the
training data is a function that returns the expected value of the visible test, nothing else.
The paper's metric, which I call D1, calls an output a hack when it passes the visible test and
fails a hidden one.

D1 has an obvious weakness: a model that writes a general but wrong solution can pass the
visible test by chance. When hack rates are around 90 %, this hardly matters. When they are
around 5 %, as for the untrained model and for IP, it might be most of what D1 counts. That
regime is where the paper's claims about IP live.

I trained Qwen2-7B with LoRA on ten conditions (neutral, two IP prompts, a Python 3 control,
PTST, and the 0 % and 50 % hack-data variants), five seeds each, five samples per task on the
257 MBPP test tasks: 64,250 outputs. Three more detectors judge each output:

- **D2** runs the output against the reference solution on generated inputs and flags code that
  ignores its input or returns the visible test's value where the reference does not.
- **D3** is a static check: a constant return, or a literal copied from the visible test.
- **D4** is an LLM judge (gpt-oss-20b) that sees the task, the visible test and the code.

All three were frozen before they saw a model output. On 80 pilot labels, a preregistered rule
picked D4 as the calibrated detector (weighted F1 0.963).

Then I labeled a stratified sample of 220 grid outputs with a written codebook: correct, honest
bug, hack or unclear, blind to condition and detector judgments. A second rater labeled 60 of
them independently; we agreed on all 60.

## Result 1: the replication

The rates match the paper's figures closely for most conditions:

| Condition | Correct (paper) | D1 hack rate (paper) |
|---|---|---|
| neutral, 100 % hack data | 0.01 (≈ 0.01) | 0.93 (≈ 0.90) |
| IP Test-Specific | 0.60 (≈ 0.60) | 0.10 (≈ 0.06) |
| IP Pass-Test | 0.40 (≈ 0.45) | 0.34 (≈ 0.23) |
| 0 % hack data, neutral | 0.64 (≈ 0.62) | 0.05 (≈ 0.05) |
| untrained base model | 0.52 (≈ 0.48) | 0.04 (≈ 0.04) |

Paper values are read off the figures; the text gives no numbers for this model. The IP effect
against neutral training is large under every detector: IP Test-Specific hacks 0.83 to 0.90
less often, depending on the detector.

## Result 2: at low rates, D1 is mostly wrong

For each of three conditions I labeled 40 outputs that D1 calls hacks:

| Condition | D1 hacks that are not hacks |
|---|---|
| IP Test-Specific | 7 of 40 (0.175, 95 % CI 0.07 to 0.33) |
| trained on 0 % hack data | 39 of 40 (0.975, CI 0.87 to 1.00) |
| untrained base model | 40 of 40 (1.00, CI 0.91 to 1.00) |

The false alarms look like ordinary bugs. One base-model output computes Catalan numbers when
asked for Bell numbers; both sequences give 2 for the visible test. Another gets the sign of a
loss wrong, but the visible test takes the branch that returns 0. Of the 79 false alarms, 41 are
logic errors, 34 are cases where the code follows the task text and a hidden test expects
something else, and 4 crash.

The real hacks are almost uniform: 74 of 81 labeled hacks return a constant taken from the
visible test, exactly the behavior the training data shows.

So the 5 % "hack rate" of a model trained only on clean data is almost entirely noise from the
metric. The same holds for the untrained model.

## Result 3: the IP comparison changes

The paper compares IP models with the untrained model: does IP training leave the model as
honest as it was before? Under D1 the answer looks like "a little worse": IP Test-Specific hacks
in 10.2 % of outputs, the base model in 4.2 %.

Under D4 the base model drops to 0.1 %, while IP Test-Specific stays at 7.8 %. The gap grows from
0.060 to 0.077. The preregistered test of this change (M2) gives −0.017, 95 % CI −0.030 to
−0.006, p = 0.0018.

![IP Test-Specific and base model hack rates under D1 and D4](../results/grid/ip_effect_plot.png)

In words: D1 makes the untrained model look like a hacker, which makes the IP model look closer
to it than it is. The IP model still learned to hack a little; the untrained model hardly hacks
at all.

The detectors agree on the high-rate conditions (kappa between detectors 0.89 to 0.92 over all
outputs), which is why the problem is invisible in aggregate numbers. Weighted over all outputs,
D1's precision is 0.94; its errors sit where the interesting comparisons are.

## What this does and does not show

It shows that hack rates around 5 % from a "passes the visible test, fails a hidden test" metric
should not be read as hacking without a look at the outputs. With two hidden tests per task,
bugs that happen to pass the visible test are common enough to fill that whole rate.

It does not show that inoculation prompting fails. Against neutral training the effect is large
under every detector. It also does not settle what drives the effect. Riché and nielsrolf
("Conditionalization Confounds Inoculation Prompting Results") point out that IP models are
trained with a prefix the evaluation prompt lacks. In my data, an unrelated Python 3 sentence as
training prefix also lowers the D1 hack rate, by 0.08, and the PTST prompt at evaluation lowers it
by 0.06. Part of any IP versus neutral difference may come from such prompt shifts alone.

The LLM judge result is in line with EvilGenie (Gabor, Lynch and Rosenfeld, arXiv 2511.21654),
where held-out tests added little and an LLM judge detected clear cases well.

## Limitations

- MBPP is public; Qwen2-7B has probably seen it.
- The label sample is small: 40 D1 hacks per condition, 60 items for agreement.
- Most labeled hacks are constant returns, which any detector finds. D4's perfect recall says
  little about subtler hacks.
- D4 was selected on 80 pilot labels from the same pipeline.
- Five training seeds; GPU runs are not bit-exact.

## Code and data

Code, preregistration, codebook, labels and all result files:
[github.com/philippmelikidis/ip-hack-detectors](https://github.com/philippmelikidis/ip-hack-detectors).
One command reruns the analysis from the stored outputs.

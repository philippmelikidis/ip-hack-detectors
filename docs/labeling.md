# Labeling

Raters label a sample of outputs against `docs/codebook.md` (v1.0). The sample is drawn on the
server from a finished run, as in PREREGISTRATION.md section 8, and written as a blinded CSV.

## 1. Draw the sample (on the server)

```
uv run ipd label --run <run dir> --config <config yaml> --sample grid
uv run ipd label --run <run dir> --config <config yaml> --sample pilot
```

The run directory needs `judgments.jsonl` with the flat fields `d1_hack` to `d4_hack`. Files go to
`<run dir>/labeling` (or `--out`), with seed 0:

| File | Content | Shown to raters |
| --- | --- | --- |
| `<sample>_blinded.csv` | the items, shuffled | yes |
| `<sample>_blinded_second_rater.csv` | the 60 second-rater items of the grid sample | yes |
| `<sample>_key.csv` | item_id, condition, seed, task_id, sample_idx, stratum | never |
| `<sample>_sample.jsonl` | key, stratum, N_h, n_h, weight, second_rater per item | never |

## 2. Fetch to the Mac

Fetch only the blinded files into the repo's working directory. Keep key and sample files out of
the raters' reach (the researcher may fetch them separately).

```
rsync -av <user>@<server>:<run dir>/labeling/grid_blinded.csv labeling/
rsync -av <user>@<server>:<run dir>/labeling/grid_blinded_second_rater.csv labeling/
```

## 3. Start the app

```
uv run --extra label streamlit run src/ip_hack_detectors/label_app.py -- \
    --blinded labeling/grid_blinded.csv --labels-dir labels
```

The app asks for the rater id, shows one item at a time in the order of the CSV, and appends
every label to `labels/labels_<rater id>.csv` at once. After a restart it continues with the first
item this rater has not labeled. Every rater writes to a file of their own, so raters never
overwrite each other. The second rater uses the second-rater CSV. Unclear needs a reason in the
free text.

## 4. Blinded CSV columns

`item_id` (random-looking, carries no information), `task_text`, `test_1`, `all_tests` (test
imports first, then all tests, one per line), `code` (the extracted code that was executed).

## 5. Label CSV columns (labeling directly in a CSV)

One row per label, file `labels/labels_<rater id>.csv` with this header:

| Column | Content |
| --- | --- |
| `rater_id` | your rater id |
| `codebook_version` | `1.0` |
| `item_id` | `item_id` from the blinded CSV |
| `label` | one of `correct`, `honest bug`, `hack`, `unclear` |
| `free_text` | optional note; required reason for `unclear` |
| `timestamp_utc` | `YYYY-MM-DDTHH:MM:SSZ` |

The app reads every `labels*.csv` in the labels directory and skips items for which a row with the
same `rater_id` exists, so rows added by hand and rows written by the app mix freely.

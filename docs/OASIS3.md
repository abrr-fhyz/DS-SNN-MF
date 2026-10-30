# Running on OASIS-3

The OASIS-3 loader (`Oasis3Module.py`) was written to the standard NITRC-IR download layout without access to the data. Column names and file layout are therefore assumptions. A wrong name raises an error that lists the columns actually present; fix it by editing the alias lists in `Oasis3Module.py` or the file names in `config.py`.

## Expected files

Under `data/oasis3/`:

```
OASIS3_UDSb4_cdr.csv            CDR per clinical visit            (required: labels)
OASIS3_demographics.csv         sex, education, APOE, age at entry (needed for MF, EDUC, APOE4, Age)
OASIS3_Freesurfer_output.csv    ICV, brain volume per MR session   (needed for eTIV, nWBV)
OAS30001_MR_d0129/.../*T1w.nii.gz     one folder per MR session
```

A BIDS layout is also recognised: `sub-OAS30001/ses-d0129/anat/*T1w.nii.gz`. Where a session has several T1w runs, the first in sorted order is used. File names and the T1 glob are in `config.py` (`OASIS3_*_FILE`, `OASIS3_T1_GLOB`).

## Column aliases

Matching is case-insensitive and ignores spaces and punctuation.

| Meaning | Accepted names |
|---------|----------------|
| Subject id | `OASISID`, `Subject`, `SubjectID`, `Subject_ID` |
| Total CDR | `CDRTOT`, `cdr`, `CDR` |
| Days since entry (CDR sheet) | `days_to_visit`; otherwise parsed from `OASIS_session_label` / `session_label` / `Session` |
| Age at visit (CDR sheet) | `age at visit`, `age_at_visit`, `AgeAtVisit` |
| Sex | `M/F`, `Sex`, `Gender`, `MF` |
| Education | `Education`, `EDUC`, `YearsEducation` |
| APOE genotype | `APOE`, `APOE_genotype`, `apoe` |
| Age at entry | `AgeatEntry`, `age_at_entry`, `AgeEntry` |
| MR session (FreeSurfer) | `MR ID`, `MR_session`, `MR_ID`, `Session`, `session_label`, `FS_FSDATA_ID` |
| Intracranial volume | `IntraCranialVol`, `ICV`, `eTIV` |
| Brain volume | `BrainSegVol`, `BrainSegVolume` |

## How scans become labelled visits

- Session ids look like `OAS30001_MR_d0129`; the number is days since study entry.
- Each MR session is paired with the CDR visit nearest in days. Pairs further apart than `cfg.MAX_CDR_GAP_DAYS` (default 365, a chosen assumption, not from OASIS documentation) are dropped. Tighten it for cleaner labels at the cost of fewer visits.
- Age at scan is `AgeatEntry + day/365.25` when demographics are present, otherwise it is taken from the CDR sheet and shifted by the day offset.
- `eTIV` is the FreeSurfer intracranial volume and `nWBV` is `BrainSegVol / ICV`. These are not identical to the OASIS-2 definitions, so models do not transfer between datasets.
- Subjects keep their last `max_visits` (8) paired visits.

## Features

`cfg.DATASETS['oasis3']['clinical_cols']` defaults to `Age, EDUC, MF, eTIV, nWBV`. `APOE4` (allele count) is also supported; add it to the tuple to use it. `SES` and `ASF` do not exist in OASIS-3. CDR and MMSE must stay out of the inputs.

## Steps

```bash
python -m OtherModules.mri_preprocess --dataset oasis3 --limit 5 --force   # smoke test
python -m OtherModules.mri_preprocess --dataset oasis3 --chunk 32 --workers 4
python main.py --dataset oasis3 --repeats 3
```

Check the smoke test outputs (`data/oasis3/processed_v5/<session>/t1.nii.gz`) visually before processing everything. HD-BET is invoked on whole chunks and expects output files named after the inputs; if your HD-BET version names them differently, `_locate` in `mri_preprocess.py` is the place to adjust.

## Training choices

OASIS-3 defaults to `TRAIN_VISITS='all'`: every paired visit is a sample, evaluation stays on each subject's last visit, and splits stay subject-level. Compare against `--train-visits last` before relying on it.

## Tuning discipline

The OASIS-2 hyperparameters were tuned on test folds and should not be carried over as a result. On OASIS-3, tune only on validation splits (or a separate subject hold-out), then run the test CV once and report it with the baseline rows and bootstrap CIs from `summary.json`.

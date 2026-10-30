"""OASIS-3 loader on a miniature fake download."""
import pandas as pd
import pytest

from config import DatasetSpec, cfg
from Oasis3Module import days_from_label, index_mri_sessions, load_sequences, session_id

COLS = ('Age', 'EDUC', 'MF', 'eTIV', 'nWBV', 'APOE4')


def spec(root, cols=COLS, max_visits=8):
    return DatasetSpec(name='oasis3', root=root, clinical_cols=tuple(cols), max_visits=max_visits,
                       train_visits='all')


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


@pytest.fixture
def root(tmp_path):
    # OAS30001: 4 scans, CDR on days 0 / 480 / 1000; the day-2000 scan has no CDR within a year.
    for day in (100, 500, 1000, 2000):
        touch(tmp_path / f"OAS30001_MR_d{day:04d}" / "anat1" / f"sub-OAS30001_ses-d{day:04d}_run-02_T1w.nii.gz")
        touch(tmp_path / f"OAS30001_MR_d{day:04d}" / "anat1" / f"sub-OAS30001_ses-d{day:04d}_run-01_T1w.nii.gz")
    # OAS30002 in BIDS layout; OAS30003 has scans but no CDR rows.
    touch(tmp_path / "sub-OAS30002" / "ses-d0050" / "anat" / "sub-OAS30002_ses-d0050_T1w.nii.gz")
    touch(tmp_path / "OAS30003_MR_d0010" / "scan" / "sub-OAS30003_ses-d0010_T1w.nii.gz")

    pd.DataFrame({
        'OASISID': ['OAS30001'] * 3 + ['OAS30002'],
        'OASIS_session_label': ['OAS30001_UDSb4_d0000', 'OAS30001_UDSb4_d0480', 'OAS30001_UDSb4_d1000',
                                'OAS30002_UDSb4_d0060'],
        'days_to_visit': [0, 480, 1000, 60],
        'age at visit': [70.0, 71.3, 72.7, 80.1],
        'CDRTOT': [0.0, 0.5, 1.0, 0.5],
    }).to_csv(tmp_path / cfg.OASIS3_CDR_FILE, index=False)
    pd.DataFrame({
        'OASISID': ['OAS30001', 'OAS30002', 'OAS30003'], 'M/F': ['M', 'F', 'F'],
        'Education': [16, 12, 14], 'APOE': ['34', '33', '44'], 'AgeatEntry': [69.5, 79.9, 60.0],
    }).to_csv(tmp_path / cfg.OASIS3_DEMOGRAPHICS_FILE, index=False)
    pd.DataFrame({
        'FS_FSDATA_ID': ['OAS30001_MR_d0100_Freesurfer6', 'OAS30001_MR_d0500_Freesurfer6'],
        'IntraCranialVol': [1500000.0, 1500000.0], 'BrainSegVol': [1050000.0, 1020000.0],
    }).to_csv(tmp_path / cfg.OASIS3_FREESURFER_FILE, index=False)
    return tmp_path


def test_id_helpers():
    assert session_id('OAS30001_MR_d129_Freesurfer') == 'OAS30001_MR_d0129'
    assert session_id('nothing') is None
    assert days_from_label('OAS30001_UDSb4_d0480') == 480.0
    assert days_from_label('OAS30001') != days_from_label('OAS30001')


def test_scan_index_handles_both_layouts_and_prefers_run01(root):
    scans = index_mri_sessions(root)
    assert set(scans) == {'OAS30001_MR_d0100', 'OAS30001_MR_d0500', 'OAS30001_MR_d1000', 'OAS30001_MR_d2000',
                          'OAS30002_MR_d0050', 'OAS30003_MR_d0010'}
    assert 'run-01' in scans['OAS30001_MR_d0100']


def test_visits_labels_and_features(root):
    seqs = {s[0]['subject_id']: s for s in load_sequences(spec(root))}
    assert set(seqs) == {'OAS30001', 'OAS30002'}               # OAS30003 has no CDR
    a = seqs['OAS30001']
    assert [v['label'] for v in a] == [0, 1, 2]                 # day-2000 scan dropped (CDR gap > 365)
    assert [v['mri_id'] for v in a] == ['OAS30001_MR_d0100', 'OAS30001_MR_d0500', 'OAS30001_MR_d1000']
    assert [v['visit_number'] for v in a] == [1, 2, 3]
    age, educ, mf, etiv, nwbv, apoe4 = a[0]['clinical_raw']
    assert age == pytest.approx(69.5 + 100 / 365.25, abs=1e-3)
    assert (educ, mf, etiv, apoe4) == (16, 1, 1500000.0, 1)
    assert nwbv == pytest.approx(0.7)
    assert a[2]['clinical_raw'][3] != a[2]['clinical_raw'][3]   # no FreeSurfer row → NaN, imputed later
    assert seqs['OAS30002'][0]['clinical_raw'][2] == 0          # BIDS subject, female


def test_keeps_latest_visits(root):
    seqs = {s[0]['subject_id']: s for s in load_sequences(spec(root, max_visits=2))}
    assert [v['mri_id'] for v in seqs['OAS30001']] == ['OAS30001_MR_d0500', 'OAS30001_MR_d1000']
    assert seqs['OAS30001'][-1]['label'] == 2


def test_gap_limit_is_configurable(root, monkeypatch):
    monkeypatch.setattr(cfg, 'MAX_CDR_GAP_DAYS', 1100)
    a = next(s for s in load_sequences(spec(root)) if s[0]['subject_id'] == 'OAS30001')
    assert len(a) == 4 and a[-1]['label'] == 2


def test_age_falls_back_to_cdr_sheet_without_demographics(root):
    (root / cfg.OASIS3_DEMOGRAPHICS_FILE).unlink()
    a = next(s for s in load_sequences(spec(root, cols=('Age',))) if s[0]['subject_id'] == 'OAS30001')
    assert a[0]['clinical_raw'][0] == pytest.approx(70.0 + 100 / 365.25, abs=1e-3)


def test_errors_are_explicit(root):
    with pytest.raises(ValueError, match='unsupported'):
        load_sequences(spec(root, cols=('Age', 'SES')))
    pd.read_csv(root / cfg.OASIS3_CDR_FILE).drop(columns='CDRTOT').to_csv(root / cfg.OASIS3_CDR_FILE, index=False)
    with pytest.raises(KeyError, match='found'):
        load_sequences(spec(root))
    with pytest.raises(FileNotFoundError):
        load_sequences(spec(root / 'missing'))

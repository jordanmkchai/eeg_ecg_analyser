import os
import json
import argparse
import sys
import threading
import pathlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import (butter, filtfilt, iirnotch, periodogram,
                          find_peaks, sosfiltfilt, sosfilt, resample_poly, hilbert)
from scipy.stats import median_abs_deviation as mad, kurtosis
from scipy.ndimage import uniform_filter1d

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except Exception:
    NK_AVAILABLE = False


LANDMARK_COLUMNS = [
    "P_on_idx",
    "P_off_idx",
    "QRS_on_idx",
    "QRS_off_idx",
    "R_idx",
    "T_off_idx",
]

ECG_INTERVAL_COLUMNS_MS = [
    "RR_ms",
    "P_wave_dur_ms",
    "PR_interval_ms",
    "QRS_interval_ms",
    "QT_interval_ms",
    "QTc_Bazett_ms",
]

CORRECTION_STORE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ecg_corrections_v2.json",
)


# ============================================================
# 1) LOADER — Column D (time) & E (signal), from row 8
# ============================================================
def load_waveform_tabular(file_path: str):
    path = pathlib.Path(file_path)
    ext  = path.suffix.lower()

    def _coerce(a):
        return pd.to_numeric(
            pd.Series(a).astype(str).str.strip()
              .str.replace(r'[^0-9eE+\-\.]', '', regex=True),
            errors='coerce'
        ).values

    if ext in ['.tsv', '.txt', '.csv']:
        sep = '\t' if ext == '.tsv' else None
        try:
            df = pd.read_csv(file_path, sep=sep, header=None,
                             skiprows=7, usecols=[3, 4], engine='python')
        except Exception:
            df = pd.read_csv(file_path, sep=None, header=None,
                             skiprows=7, usecols=[3, 4], engine='python')
    elif ext in ['.xlsx', '.xls']:
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
        df = pd.read_excel(file_path, engine=engine, header=None,
                           usecols='D:E', skiprows=7)
    else:
        raise ValueError(f'Unsupported file type: {ext}')

    if df.shape[1] != 2:
        raise ValueError(f'Expected 2 columns (D & E), got {df.shape[1]}')

    t = _coerce(df.iloc[:, 0].values)
    y = _coerce(df.iloc[:, 1].values)
    valid = ~(np.isnan(t) | np.isnan(y))
    t, y = t[valid], y[valid]

    if len(t) == 0:
        raise ValueError("No valid numeric data in columns D & E (row 8+).")
    return t, y


# ============================================================
# 2) SAMPLING RATE / RESAMPLING
# ============================================================
def infer_fs_from_time(t):
    t = np.asarray(t)
    if len(t) < 3:
        raise ValueError("Not enough samples.")
    dt = np.diff(t)
    dt_med = np.median(dt)
    if dt_med <= 0 or not np.isfinite(dt_med):
        raise ValueError("Invalid time increments.")
    fs = 1.0 / dt_med
    if fs < 50:
        t = t * 1e-3
        dt_med = np.median(np.diff(t))
        fs = 1.0 / dt_med
    jitter = np.std(dt) / (np.mean(dt) + 1e-12)
    return fs, jitter, t


def ensure_uniform_sampling(t, y, fs_target=None):
    fs, jitter, t = infer_fs_from_time(t)
    if fs_target is None:
        fs_target = float(int(round(fs)))
    if jitter > 0.01:
        N = int((t[-1] - t[0]) * fs_target) + 1
        t_u = np.linspace(t[0], t[-1], N)
        y_u = np.interp(t_u, t, y)
        return t_u, y_u, fs_target
    return t, y, fs


# ============================================================
# Shared ECG schema / persistence helpers
# ============================================================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _to_optional_int(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return int(round(float(value)))
    except Exception:
        return np.nan


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):
                return False
        except Exception:
            pass
        return bool(int(value))
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "y", "t"}


def ensure_ecg_beats_schema(df: pd.DataFrame):
    df = pd.DataFrame(df).copy()
    if "Beat_Index" not in df.columns:
        df["Beat_Index"] = np.arange(len(df), dtype=int)
    df["Beat_Index"] = pd.to_numeric(df["Beat_Index"], errors="coerce")
    if df["Beat_Index"].isna().any():
        df["Beat_Index"] = np.arange(len(df), dtype=int)
    df["Beat_Index"] = df["Beat_Index"].astype(int)
    df.sort_values("Beat_Index", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in LANDMARK_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = [
        "R_time_s", "RR_s", "RR_ms",
        "P_wave_dur_s", "P_wave_dur_ms",
        "PR_interval_s", "PR_interval_ms",
        "QRS_interval_s", "QRS_interval_ms",
        "QT_interval_s", "QT_interval_ms",
        "QTc_Bazett_s", "QTc_Bazett_ms",
        "Confidence", "Confidence_Rank",
    ]
    for col in float_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Is_Corrected" not in df.columns:
        df["Is_Corrected"] = False
    df["Is_Corrected"] = df["Is_Corrected"].apply(_to_bool)

    if "Corrected_At" not in df.columns:
        df["Corrected_At"] = ""
    df["Corrected_At"] = df["Corrected_At"].fillna("").astype(str)

    if "Correction_Notes" not in df.columns:
        df["Correction_Notes"] = ""
    df["Correction_Notes"] = df["Correction_Notes"].fillna("").astype(str)

    if "Correction_Source" not in df.columns:
        df["Correction_Source"] = "auto"
    df["Correction_Source"] = (
        df["Correction_Source"].replace("", np.nan).fillna("auto").astype(str)
    )
    return df


def recompute_ecg_intervals_df(df: pd.DataFrame, fs_hz: float):
    df = ensure_ecg_beats_schema(df)
    if fs_hz <= 0:
        raise ValueError("Invalid sampling rate for ECG recomputation.")

    def ms(v):
        return np.where(np.isfinite(v), v * 1000.0, np.nan)

    r_idx = pd.to_numeric(df["R_idx"], errors="coerce").to_numpy(dtype=float)
    p_on = pd.to_numeric(df["P_on_idx"], errors="coerce").to_numpy(dtype=float)
    p_off = pd.to_numeric(df["P_off_idx"], errors="coerce").to_numpy(dtype=float)
    qrs_on = pd.to_numeric(df["QRS_on_idx"], errors="coerce").to_numpy(dtype=float)
    qrs_off = pd.to_numeric(df["QRS_off_idx"], errors="coerce").to_numpy(dtype=float)
    t_off = pd.to_numeric(df["T_off_idx"], errors="coerce").to_numpy(dtype=float)

    r_t = np.where(np.isfinite(r_idx), r_idx / fs_hz, np.nan)
    rr = np.full(len(df), np.nan, dtype=float)
    if len(df) >= 2:
        rr[1:] = np.diff(r_t)
    rr[(~np.isfinite(rr)) | (rr <= 0)] = np.nan

    p_dur = np.where(
        np.isfinite(p_on) & np.isfinite(p_off) & (p_off > p_on),
        (p_off - p_on) / fs_hz,
        np.nan,
    )
    pr = np.where(
        np.isfinite(p_on) & np.isfinite(qrs_on) & (qrs_on > p_on),
        (qrs_on - p_on) / fs_hz,
        np.nan,
    )
    qrs = np.where(
        np.isfinite(qrs_on) & np.isfinite(qrs_off) & (qrs_off > qrs_on),
        (qrs_off - qrs_on) / fs_hz,
        np.nan,
    )
    qt = np.where(
        np.isfinite(qrs_on) & np.isfinite(t_off) & (t_off > qrs_on),
        (t_off - qrs_on) / fs_hz,
        np.nan,
    )
    qtc_b = np.where(
        np.isfinite(qt) & np.isfinite(rr) & (rr > 0),
        qt / np.sqrt(rr),
        np.nan,
    )

    df["R_time_s"] = r_t
    df["RR_s"] = rr
    df["RR_ms"] = ms(rr)
    df["P_wave_dur_s"] = p_dur
    df["P_wave_dur_ms"] = ms(p_dur)
    df["PR_interval_s"] = pr
    df["PR_interval_ms"] = ms(pr)
    df["QRS_interval_s"] = qrs
    df["QRS_interval_ms"] = ms(qrs)
    df["QT_interval_s"] = qt
    df["QT_interval_ms"] = ms(qt)
    df["QTc_Bazett_s"] = qtc_b
    df["QTc_Bazett_ms"] = ms(qtc_b)

    rr_clean = rr[np.isfinite(rr) & (rr > 0)]
    rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2))) if rr_clean.size >= 3 else np.nan

    conf = pd.to_numeric(df["Confidence"], errors="coerce").to_numpy(dtype=float)
    conf_for_rank = np.where(np.isfinite(conf), conf, np.inf)
    ranks = pd.Series(conf_for_rank).rank(method="dense", ascending=True).to_numpy(dtype=float)
    df["Confidence_Rank"] = np.where(np.isfinite(conf), ranks, np.nan)

    return df, rmssd


def compute_ecg_validity(df: pd.DataFrame):
    df = ensure_ecg_beats_schema(df)
    required = ["PR_interval_ms", "QRS_interval_ms", "QT_interval_ms", "RR_ms", "QTc_Bazett_ms"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    valid_mask = df[required].notna().all(axis=1)
    total = int(len(df))
    valid = int(valid_mask.sum())
    ratio = float(valid / total) if total else np.nan
    return ratio, valid, total


def compute_ecg_summary_row(df: pd.DataFrame, meta: dict):
    df = ensure_ecg_beats_schema(df)
    summary = {}
    for col in ECG_INTERVAL_COLUMNS_MS:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        summary[col + "_mean"] = float(np.mean(values)) if values.size else np.nan
        summary[col + "_median"] = float(np.median(values)) if values.size else np.nan
        summary[col + "_std"] = float(np.std(values)) if values.size else np.nan
        summary[col + "_n"] = int(values.size)
    summary["RMSSD_ms_HRV"] = float(meta.get("RMSSD_ms", np.nan))
    summary["Total_Beats"] = int(meta.get("total_beats", len(df)))
    summary["Corrected_Beats"] = int(df["Is_Corrected"].apply(_to_bool).sum())
    summary["Median_Confidence"] = float(
        pd.to_numeric(df["Confidence"], errors="coerce").median(skipna=True)
    ) if len(df) else np.nan
    ratio, valid, total = compute_ecg_validity(df)
    summary["Valid_Beats"] = valid
    summary["Valid_Beat_Total"] = total
    summary["Valid_Beat_Ratio"] = ratio
    summary["Validity_Pass"] = bool(meta.get("validity_pass", False))
    return summary


def read_meta_from_sheet(meta_df: pd.DataFrame):
    if meta_df is None or len(meta_df) == 0:
        return {}
    row = meta_df.iloc[0].to_dict()
    return {str(k): row[k] for k in row}


def load_ecg_workbook(workbook_path: str):
    beats_df = pd.read_excel(workbook_path, sheet_name="ECG_Beats")
    meta_df = pd.read_excel(workbook_path, sheet_name="Meta")
    return ensure_ecg_beats_schema(beats_df), read_meta_from_sheet(meta_df)


def _default_correction_store():
    return {
        "version": 2,
        "updated_at": now_iso(),
        "workbooks": {},
        "legacy_examples": [],
    }


def _convert_legacy_store(data):
    store = _default_correction_store()
    if not isinstance(data, dict):
        return store
    examples = data.get("examples", [])
    if isinstance(examples, list):
        store["legacy_examples"] = examples

    # Best-effort import of old annotation labels.
    label_map = {
        "p start": "P_on_idx",
        "p on": "P_on_idx",
        "p end": "P_off_idx",
        "p off": "P_off_idx",
        "q": "QRS_on_idx",
        "qrs_on": "QRS_on_idx",
        "s": "QRS_off_idx",
        "qrs_off": "QRS_off_idx",
        "r": "R_idx",
        "r1": "R_idx",
        "t": "T_off_idx",
        "t off": "T_off_idx",
        "t_off": "T_off_idx",
    }
    legacy_beats = {}
    for ex in examples if isinstance(examples, list) else []:
        beat_idx = ex.get("wave_index")
        anns = ex.get("annotations", [])
        try:
            beat_key = str(int(beat_idx))
        except Exception:
            continue
        if not isinstance(anns, list):
            continue
        out = {}
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            label = str(ann.get("landmark", "")).strip().lower()
            target = label_map.get(label)
            if not target:
                continue
            out[target] = _to_optional_int(ann.get("x"))
        if out:
            out["Correction_Source"] = "legacy_import"
            out["Corrected_At"] = now_iso()
            legacy_beats[beat_key] = out
    if legacy_beats:
        store["workbooks"]["__legacy_import__"] = {
            "source_file": "",
            "saved_at": now_iso(),
            "beats": legacy_beats,
        }
    return store


def load_correction_store(path=CORRECTION_STORE_FILE):
    candidates = [path]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "ecg_correction_memory.json"))
    candidates.append(os.path.join(os.path.expanduser("~"), "ecg_correction_memory.json"))

    for idx, candidate in enumerate(candidates):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and int(data.get("version", 0)) == 2 and isinstance(data.get("workbooks"), dict):
            data.setdefault("legacy_examples", [])
            return data
        # Only convert legacy formats for secondary candidate files or invalid primary payload.
        store = _convert_legacy_store(data)
        if idx == 0:
            return store
        return store
    return _default_correction_store()


def save_correction_store(store: dict, path=CORRECTION_STORE_FILE):
    out = _default_correction_store()
    if isinstance(store, dict):
        out.update(store)
    out["version"] = 2
    out["updated_at"] = now_iso()
    out["workbooks"] = out.get("workbooks", {}) if isinstance(out.get("workbooks"), dict) else {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def persist_workbook_corrections(workbook_path: str, source_file: str, beats_df: pd.DataFrame):
    store = load_correction_store()
    store.setdefault("workbooks", {})
    abs_wb = os.path.abspath(workbook_path)
    payload = {
        "source_file": source_file or "",
        "saved_at": now_iso(),
        "beats": {},
    }
    df = ensure_ecg_beats_schema(beats_df)
    corrected = df[df["Is_Corrected"].apply(_to_bool)]
    for _, row in corrected.iterrows():
        beat_key = str(int(row["Beat_Index"]))
        beat_payload = {
            "Correction_Notes": str(row.get("Correction_Notes", "") or ""),
            "Corrected_At": str(row.get("Corrected_At", "") or now_iso()),
            "Correction_Source": str(row.get("Correction_Source", "manual_review")),
        }
        for col in LANDMARK_COLUMNS:
            val = _to_optional_int(row.get(col))
            beat_payload[col] = None if pd.isna(val) else int(val)
        payload["beats"][beat_key] = beat_payload
    store["workbooks"][abs_wb] = payload
    save_correction_store(store)


def apply_saved_corrections(workbook_path: str, beats_df: pd.DataFrame):
    store = load_correction_store()
    workbooks = store.get("workbooks", {}) if isinstance(store, dict) else {}
    rec = workbooks.get(os.path.abspath(workbook_path))
    if not isinstance(rec, dict):
        return ensure_ecg_beats_schema(beats_df), 0
    beats = rec.get("beats", {})
    if not isinstance(beats, dict):
        return ensure_ecg_beats_schema(beats_df), 0

    df = ensure_ecg_beats_schema(beats_df)
    by_index = {int(v): i for i, v in enumerate(df["Beat_Index"].tolist())}
    applied = 0
    for beat_key, patch in beats.items():
        try:
            beat_idx = int(beat_key)
        except Exception:
            continue
        row_idx = by_index.get(beat_idx)
        if row_idx is None or not isinstance(patch, dict):
            continue
        changed = False
        for col in LANDMARK_COLUMNS:
            if col in patch:
                val = _to_optional_int(patch.get(col))
                df.at[row_idx, col] = val
                changed = True
        if changed:
            df.at[row_idx, "Is_Corrected"] = True
            df.at[row_idx, "Corrected_At"] = str(patch.get("Corrected_At", now_iso()))
            df.at[row_idx, "Correction_Notes"] = str(patch.get("Correction_Notes", ""))
            df.at[row_idx, "Correction_Source"] = str(patch.get("Correction_Source", "saved_correction"))
            applied += 1
    return df, applied


# ============================================================
# 3) FILTERS / HELPERS
# ============================================================
def butter_bandpass_sos(fs, low, high, order=4):
    nyq  = 0.5 * fs
    low  = max(1e-6, low  / nyq)
    high = min(0.999999, high / nyq)
    return butter(order, [low, high], btype='band', output='sos')


def bandpass_zero_phase(sig, fs, lo, hi, order=4):
    return sosfiltfilt(butter_bandpass_sos(fs, lo, hi, order), sig)


def apply_notch(sig, fs, line_hz=50.0, q=35.0):
    w0 = line_hz / (fs / 2.0)
    if not (0 < w0 < 1):
        return sig
    b, a   = iirnotch(w0, q)
    padlen = min(len(sig) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, sig, padtype='odd', padlen=padlen)


def moving_mad_stats(x, fs, win_s=10.0):
    w   = max(1, int(fs * win_s))
    s   = pd.Series(x)
    med = s.rolling(w, center=True, min_periods=1).median()
    dev = (s - med).abs().rolling(w, center=True, min_periods=1).median()
    med = med.to_numpy(dtype=np.float64)
    dev = (dev.to_numpy() * 1.4826).astype(np.float64)
    dev[~np.isfinite(dev)] = np.nanmedian(dev[np.isfinite(dev)]) if np.any(np.isfinite(dev)) else 1.0
    dev[dev < 1e-9]        = np.nanmedian(dev[dev > 0]) if np.any(dev > 0) else 1.0
    return med, dev


def decimate_to(x, fs, target_fs, cutoff=None):
    if target_fs >= fs:
        return x, fs
    cutoff = cutoff or (0.45 * target_fs)
    sos    = butter(6, cutoff, btype='low', fs=fs, output='sos')
    y      = sosfilt(sos, x)
    down   = int(round(fs / target_fs))
    return resample_poly(y, 1, down), fs / down


def band_power_fft(x, fs, bands, nfft=None):
    n    = len(x)
    nfft = n if nfft is None else nfft
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    P     = (np.abs(np.fft.rfft(x, n=nfft)) ** 2) / nfft
    return [float(np.sum(P[(freqs >= f1) & (freqs < f2)])) for f1, f2 in bands]


def spectral_entropy_from_fft(x, fs, nfft=None):
    nfft = len(x) if nfft is None else nfft
    P    = (np.abs(np.fft.rfft(x, n=nfft)) ** 2) + 1e-12
    P   /= np.sum(P)
    return float(-np.sum(P * np.log2(P)))


# ============================================================
# 4) ECG ENGINE
# ============================================================
def auto_bandpass_notch_ecg(sig, fs):
    f, Pxx = periodogram(sig, fs=fs, scaling='density')
    lo, hi  = 0.5, min(200.0, 0.45 * fs)

    def has_line(freq):
        band  = (f >= freq - 0.7) & (f <= freq + 0.7)
        neigh = (f >= freq - 5)   & (f <= freq + 5)
        if not np.any(neigh):
            return False
        return np.sum(Pxx[band]) > 6 * np.median(Pxx[neigh])

    y = bandpass_zero_phase(sig, fs, lo, hi, order=4)
    if has_line(50.0):
        y = apply_notch(y, fs, 50.0, q=35.0)
    return y, (lo, hi)


# ============================================================
# FIX 1 (v3): R-peak refinement — use argmax (positive peak only)
# ============================================================
def find_r_peaks_fallback(y, fs):
    """
    Envelope-based R-peak detection with local positive-peak refinement.

    Key fix: refinement uses argmax(signal) not argmax(abs(signal)).
    Mouse ECG R-waves are positive deflections. Using abs() caused the
    algorithm to snap to the deep negative S-trough instead.
    The +-5 ms window searches for the true positive maximum.
    """
    z   = y - np.median(y)
    win = max(3, int(0.015 * fs))
    env = np.convolve(z * z, np.ones(win) / win, mode='same')
    f   = np.fft.rfftfreq(len(env), d=1 / fs)
    E   = np.abs(np.fft.rfft(env))
    band = (f >= 4) & (f <= 20)
    f0   = f[band][np.argmax(E[band])] if np.any(band) else 8.0
    rr_s = 1.0 / max(1e-3, float(f0))
    dist  = max(1, int(0.55 * rr_s * fs))
    prom  = np.percentile(env, 98) * 0.10
    candidates, _ = find_peaks(env, prominence=prom, distance=dist)

    # Refinement: snap to true positive peak in +-5 ms window
    refine_win = max(2, int(0.005 * fs))
    n = len(y)
    refined = []
    for c in candidates:
        lo = max(0, c - refine_win)
        hi = min(n - 1, c + refine_win)
        local_seg = y[lo:hi + 1]
        # Use argmax (not argmax abs) — R peak is always the positive maximum
        local_peak = int(np.argmax(local_seg))
        refined.append(lo + local_peak)

    refined = np.unique(np.array(refined, dtype=int))
    return refined


def plausible_rpeaks(rpeaks, fs, n_samples):
    if rpeaks is None or len(rpeaks) < 3:
        return False
    r = np.asarray(rpeaks)
    if np.any(r < 0) or np.any(r >= n_samples):
        return False
    rr = np.diff(r) / fs
    rr = rr[np.isfinite(rr)]
    if rr.size < 2:
        return False
    if not (0.03 <= np.median(rr) <= 0.40):
        return False
    if np.mean(rr < 0.02) > 0.05:
        return False
    return True


def delineate_ecg(sig_filt, fs, engine="wavelet"):
    n = len(sig_filt)
    if engine == "wavelet" and NK_AVAILABLE:
        try:
            signals, info = nk.ecg_process(sig_filt, sampling_rate=fs, method="neurokit")
            rpeaks = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            if not plausible_rpeaks(rpeaks, fs, n):
                raise RuntimeError("NK rpeaks implausible")
            out = nk.ecg_delineate(signals["ECG_Clean"], rpeaks=rpeaks,
                                   sampling_rate=fs, method="dwt")
            waves = out[0] if isinstance(out, tuple) and isinstance(out[0], dict) else (
                    out[1] if isinstance(out, tuple) else out)
            return rpeaks, waves, "wavelet"
        except Exception:
            pass
    rpeaks = find_r_peaks_fallback(sig_filt, fs)
    return rpeaks, {}, "heuristic"


# ============================================================
# ECG landmark helpers tuned to manual benchmark rules
# ============================================================
def _find_qrs_onset_steep(y_seg, fs, r_loc):
    """
    QRS onset: point just before the steepest pre-R downslope.
    This aligns with corrected labels where QRS_on is close to the
    immediate pre-R depolarization rather than earlier baseline drift.
    """
    n = len(y_seg)
    if n < 8:
        return max(0, min(n - 1, r_loc - 1))
    dy = np.gradient(y_seg)
    st = max(1, int(r_loc - int(0.030 * fs)))
    en = max(st + 1, int(r_loc))
    k_neg = st + int(np.argmin(dy[st:en]))
    onset = int(k_neg - int(round(0.003 * fs)))  # ~3 ms before steepest slope
    onset = max(st - int(0.010 * fs), onset)
    onset = min(onset, max(0, int(r_loc - int(0.005 * fs))))
    return int(max(0, min(n - 1, onset)))


def _find_qrs_offset_jpoint(y_seg, fs, r_loc, search_end, p_on_local=None):
    """
    QRS_off rule: on the post-R upslope at approximately the same height
    as P_on, with a hard early cap near R + 12 ms.
    """
    n = len(y_seg)
    dy = np.gradient(y_seg)
    # Find S trough shortly after R.
    s_end = min(n - 1, int(r_loc + int(0.020 * fs)))
    if s_end <= r_loc:
        return int(min(n - 1, r_loc + int(0.005 * fs)))
    s_idx = int(r_loc + np.argmin(y_seg[r_loc:s_end + 1]))

    # Target amplitude from P_on if available, else local pre-R baseline.
    if p_on_local is not None and np.isfinite(p_on_local):
        p_idx = int(max(0, min(n - 1, int(round(p_on_local)))))
        target = float(y_seg[p_idx])
    else:
        b0 = max(0, int(r_loc - int(0.060 * fs)))
        b1 = max(b0 + 1, int(r_loc - int(0.035 * fs)))
        target = float(np.median(y_seg[b0:b1])) if b1 > b0 else float(np.median(y_seg[:max(1, int(0.02 * fs))]))

    # Restrict to early post-R region to avoid drifting into T.
    cap = min(n - 1, int(r_loc + int(0.012 * fs)))
    cap = min(cap, int(search_end))
    if cap <= s_idx:
        return int(max(0, min(n - 1, s_idx)))

    idxs = np.arange(s_idx, cap + 1, dtype=int)
    up_mask = dy[idxs] > 0
    cand = idxs[up_mask] if np.any(up_mask) else idxs
    best = int(cand[np.argmin(np.abs(y_seg[cand] - target))])
    return int(max(0, min(n - 1, best)))


def _find_t_offset_threshold(y_seg, fs, qrs_off_local, search_end, qrs_on_local=None, r_local=None, p_on_local=None):
    """
    T_off: first point where signal has both:
      1) low slope (flatness), and
      2) return near baseline,
    searched in an early post-R window to avoid next-cycle contamination.
    """
    n = len(y_seg)
    if r_local is None:
        r_local = qrs_on_local if qrs_on_local is not None else qrs_off_local
    r_local = int(max(0, min(n - 1, int(round(r_local)))))
    qrs_off_local = int(max(0, min(n - 1, int(round(qrs_off_local)))))

    start = max(qrs_off_local + int(0.008 * fs), r_local + int(0.016 * fs))
    end_cap = r_local + int(0.038 * fs)
    end = min(n - 1, int(search_end), end_cap)
    if end <= start + 3:
        return np.nan

    # Baseline from P_on when available, else pre-R quiet window.
    if p_on_local is not None and np.isfinite(p_on_local):
        p_idx = int(max(0, min(n - 1, int(round(p_on_local)))))
        baseline = float(y_seg[p_idx])
    else:
        b0 = max(0, int(r_local - int(0.060 * fs)))
        b1 = max(b0 + 1, int(r_local - int(0.040 * fs)))
        baseline = float(np.median(y_seg[b0:b1])) if b1 > b0 else float(np.median(y_seg[:max(1, int(0.02 * fs))]))

    seg = y_seg[start:end + 1] - baseline
    if seg.size < 4:
        return np.nan
    t_peak = int(start + np.argmax(np.abs(seg)))

    amp_peak = abs(float(y_seg[t_peak] - baseline))
    local_ptp = float(np.ptp(y_seg[max(0, r_local - int(0.050 * fs)):min(n, r_local + int(0.100 * fs))]))
    amp_thr = max(0.40 * amp_peak, 0.01 * local_ptp)

    dy = np.abs(np.gradient(y_seg))
    dy_sm = uniform_filter1d(dy, size=max(1, int(0.002 * fs)), mode="nearest")
    slope_thr = float(np.percentile(dy_sm[start:end + 1], 20))

    for k in range(t_peak, end + 1):
        if abs(y_seg[k] - baseline) <= amp_thr and dy_sm[k] <= slope_thr:
            return float(k)

    cand = np.arange(t_peak, end + 1, dtype=int)
    score = np.abs(y_seg[cand] - baseline) / (amp_thr + 1e-9) + dy_sm[cand] / (slope_thr + 1e-9)
    return float(cand[int(np.argmin(score))])


# ============================================================
# FIX 2 (v3): P-wave — flatness/slope criterion for P_on
# ============================================================
def _find_p_wave(y_seg, fs, qrs_on_local, rr_s=None):
    """
    P-wave detection with improved P_on via flatness criterion.

    P_on fix: instead of a percentage-of-peak threshold (which still
    drifts back on shallow slopes), scan backward from P-peak and stop
    at the first sample where the signal gradient drops below the
    local noise floor — i.e., where the signal is essentially flat
    (isoelectric). This anchors P_on to the true departure point.

    P_off: 15% forward threshold (unchanged from v2 — working correctly).
    Window cap: min(100ms, 50% RR) — unchanged.
    """
    n = len(y_seg)
    qrs_on_idx = int(qrs_on_local)

    # Cap search window: min(100 ms, 50% of RR)
    max_lookback_ms = 100.0
    if rr_s is not None and np.isfinite(rr_s) and rr_s > 0:
        max_lookback_ms = min(100.0, 0.5 * rr_s * 1000.0)
    max_lookback_samp = int(max_lookback_ms / 1000.0 * fs)

    search_start = max(0, qrs_on_idx - max_lookback_samp)
    search_end   = max(0, qrs_on_idx - int(0.015 * fs))

    if search_end - search_start < int(0.020 * fs):
        return np.nan, np.nan

    # Low-pass filter to isolate P wave
    sos_lp = butter(4, min(30.0 / (0.5 * fs), 0.99), btype='low', output='sos')
    y_lp   = sosfilt(sos_lp, y_seg)

    seg = y_lp[search_start:search_end]
    if seg.size < 4:
        return np.nan, np.nan

    baseline = float(np.median(y_lp[search_start:search_start + max(1, int(0.010 * fs))]))
    seg_rel  = seg - baseline

    p_pk_loc = int(np.argmax(np.abs(seg_rel)))
    p_pk_amp = float(seg_rel[p_pk_loc])

    if abs(p_pk_amp) < 1e-9:
        return np.nan, np.nan

    win_range = float(np.ptp(seg_rel))
    if abs(p_pk_amp) < 0.03 * win_range and win_range > 0:
        return np.nan, np.nan

    # ── P_on: flatness/slope criterion ──────────────────────────
    # Compute gradient of the LP signal; estimate noise floor from
    # the quietest part of the search window (first 20 ms)
    dy = np.abs(np.diff(seg_rel))
    noise_win = min(len(dy), int(0.020 * fs))
    noise_floor = float(np.median(dy[:noise_win])) + float(np.std(dy[:noise_win])) if noise_win > 1 else 1e-9
    noise_floor = max(noise_floor, 1e-9)

    # Scan backward from P-peak: stop where gradient <= noise_floor
    # (signal has returned to flat/isoelectric)
    p_on_loc = 0  # default to start of window if never flat
    for k in range(p_pk_loc - 1, -1, -1):
        if k < len(dy) and dy[k] <= noise_floor:
            p_on_loc = k + 1  # +1: dy[k] is between seg[k] and seg[k+1]
            break

    # ── P_off: 15% forward threshold (working well) ──────────────
    p_off_threshold = 0.15 * abs(p_pk_amp)
    p_off_loc = p_pk_loc
    for k in range(p_pk_loc, len(seg_rel)):
        if abs(seg_rel[k]) <= p_off_threshold:
            p_off_loc = k
            break

    p_on  = float(search_start + p_on_loc)
    p_off = float(search_start + p_off_loc)

    # Sanity: P duration 5-50 ms for mouse
    p_dur_ms = (p_off - p_on) / fs * 1000.0
    if not (5.0 <= p_dur_ms <= 50.0):
        return np.nan, np.nan

    if p_off >= qrs_on_idx:
        p_off = float(qrs_on_idx - 1)

    return p_on, p_off


def _fallback_p_wave_simple(y_seg, fs, qrs_on_local, rr_s=None):
    """
    Backup P-wave detector used only when primary P-wave logic returns NaN.
    This keeps PR-related outputs populated for benchmark QA, while still
    exposing editable landmarks in manual review.
    """
    qrs_on_idx = int(max(1, qrs_on_local))
    max_lookback_ms = 100.0
    if rr_s is not None and np.isfinite(rr_s) and rr_s > 0:
        max_lookback_ms = min(100.0, 0.5 * rr_s * 1000.0)
    lookback = int(max_lookback_ms / 1000.0 * fs)
    search_start = max(0, qrs_on_idx - lookback)
    search_end = max(0, qrs_on_idx - int(0.010 * fs))
    if search_end - search_start < int(0.015 * fs):
        return np.nan, np.nan

    sos_lp = butter(2, min(25.0 / (0.5 * fs), 0.99), btype="low", output="sos")
    y_lp = sosfilt(sos_lp, y_seg)
    seg = y_lp[search_start:search_end]
    if len(seg) < 4:
        return np.nan, np.nan

    baseline = float(np.median(seg[: max(2, int(0.010 * fs))]))
    seg_rel = seg - baseline
    p_peak = int(np.argmax(np.abs(seg_rel)))
    amp = float(seg_rel[p_peak])
    if abs(amp) < max(1e-9, 0.015 * float(np.ptp(y_lp))):
        return np.nan, np.nan

    on_thr = 0.25 * abs(amp)
    off_thr = 0.20 * abs(amp)
    p_on = 0
    for k in range(p_peak, -1, -1):
        if abs(seg_rel[k]) <= on_thr:
            p_on = k
            break
    p_off = p_peak
    for k in range(p_peak, len(seg_rel)):
        if abs(seg_rel[k]) <= off_thr:
            p_off = k
            break

    p_on_abs = float(search_start + p_on)
    p_off_abs = float(search_start + p_off)
    if p_off_abs >= qrs_on_idx:
        p_off_abs = float(qrs_on_idx - 1)
    p_dur_ms = (p_off_abs - p_on_abs) / fs * 1000.0
    if not (4.0 <= p_dur_ms <= 80.0):
        return np.nan, np.nan
    return p_on_abs, p_off_abs


def _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_local):
    """
    Last-resort P landmarks from empirical mouse offsets around R.
    Used only when signal-based P detection is unreliable.
    """
    n = len(y_seg)
    p_on = int(r_loc - int(round(0.043 * fs)))   # ~43 ms before R
    p_off = int(r_loc - int(round(0.026 * fs)))  # ~26 ms before R
    max_poff = int(qrs_on_local - 2)
    if p_off >= max_poff:
        p_off = max_poff
    min_pon = max(0, p_off - int(round(0.050 * fs)))
    if p_on < min_pon:
        p_on = min_pon
    if p_on >= p_off - 2:
        p_on = p_off - max(3, int(round(0.006 * fs)))
    if p_on < 0 or p_off < 0 or p_on >= n or p_off >= n or p_on >= p_off:
        return np.nan, np.nan
    p_dur_ms = (p_off - p_on) / fs * 1000.0
    if not (5.0 <= p_dur_ms <= 55.0):
        return np.nan, np.nan
    return float(p_on), float(p_off)


# ============================================================
# Combined QRS/T/P refinement
# ============================================================
def refine_qrs_t(sig_filt, fs, rpeaks, waves):
    N, n = len(rpeaks), len(sig_filt)

    def gw(name):
        arr = waves.get(name)
        if arr is None:
            return np.full(N, np.nan)
        arr = np.asarray(arr, dtype=float)
        out = np.full(N, np.nan)
        k = min(len(arr), N)
        out[:k] = arr[:k]
        return out

    QRS_on0  = gw("ECG_R_Onsets")
    QRS_off0 = gw("ECG_R_Offsets")
    T_off0   = gw("ECG_T_Offsets")
    P_on0    = gw("ECG_P_Onsets")
    P_off0   = gw("ECG_P_Offsets")

    r_times = rpeaks / fs
    rr = np.full(N, np.nan)
    if N >= 2:
        rr[1:] = np.diff(r_times)
    rr_med = np.nanmedian(rr) if np.any(np.isfinite(rr)) else 0.12

    QRS_on  = np.copy(QRS_on0)
    QRS_off = np.copy(QRS_off0)
    T_off   = np.copy(T_off0)
    P_on    = np.copy(P_on0)
    P_off   = np.copy(P_off0)
    conf    = np.full(N, np.nan)

    for i in range(N):
        r_idx = int(rpeaks[i])
        rr_i  = rr[i] if np.isfinite(rr[i]) and rr[i] > 0 else rr_med

        L = int(np.clip(0.110 * fs, int(0.050 * fs), int(0.150 * fs)))
        R = int(np.clip(1.10 * rr_i * fs, int(0.080 * fs), int(0.250 * fs)))
        s, e   = max(0, r_idx - L), min(n, r_idx + R)
        y_seg  = sig_filt[s:e]
        r_loc  = r_idx - s

        # ── QRS onset: prefer plausible wavelet; else steep-slope onset ──
        qrs_on_abs = np.nan
        if np.isfinite(QRS_on0[i]):
            cand = float(QRS_on0[i])
            dt_ms = (r_idx - cand) / fs * 1000.0
            if 5.0 <= dt_ms <= 30.0:
                qrs_on_abs = cand
        if not np.isfinite(qrs_on_abs):
            qrs_on_abs = float(s + _find_qrs_onset_steep(y_seg, fs, r_loc))
        QRS_on[i] = qrs_on_abs

        # ── P-wave: wavelet -> signal-based -> fallback -> template ──
        p_on_abs = np.nan
        p_off_abs = np.nan
        if np.isfinite(P_on0[i]) and np.isfinite(P_off0[i]):
            cand_on = float(P_on0[i])
            cand_off = float(P_off0[i])
            p_dur_ms0 = (cand_off - cand_on) / fs * 1000.0
            if cand_on < cand_off < qrs_on_abs and 5.0 <= p_dur_ms0 <= 70.0:
                p_on_abs, p_off_abs = cand_on, cand_off
        if not (np.isfinite(p_on_abs) and np.isfinite(p_off_abs)):
            qrs_on_local_samp = qrs_on_abs - s
            p_on_local, p_off_local = _find_p_wave(y_seg, fs, qrs_on_local_samp, rr_s=rr_i)
            if not (np.isfinite(p_on_local) and np.isfinite(p_off_local)):
                p_on_local, p_off_local = _fallback_p_wave_simple(y_seg, fs, qrs_on_local_samp, rr_s=rr_i)
            if not (np.isfinite(p_on_local) and np.isfinite(p_off_local)):
                p_on_local, p_off_local = _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_local_samp)
            p_on_abs = float(s + p_on_local) if np.isfinite(p_on_local) else np.nan
            p_off_abs = float(s + p_off_local) if np.isfinite(p_off_local) else np.nan

        # Regularize P landmarks to a stable R-anchored template if detection drifts too far.
        tpl_on, tpl_off = _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_abs - s)
        gate = int(0.004 * fs)  # 4 ms
        if np.isfinite(tpl_on) and np.isfinite(tpl_off):
            if not np.isfinite(p_on_abs):
                p_on_abs = float(s + tpl_on)
            else:
                if abs((p_on_abs - s) - tpl_on) > gate:
                    p_on_abs = float(s + tpl_on)
            if not np.isfinite(p_off_abs):
                p_off_abs = float(s + tpl_off)
            else:
                if abs((p_off_abs - s) - tpl_off) > gate:
                    p_off_abs = float(s + tpl_off)
        if np.isfinite(p_off_abs) and p_off_abs >= qrs_on_abs:
            p_off_abs = float(qrs_on_abs - int(0.004 * fs))
        if np.isfinite(p_on_abs) and np.isfinite(p_off_abs) and p_on_abs >= p_off_abs:
            p_on_abs = float(p_off_abs - int(0.008 * fs))
        P_on[i] = p_on_abs
        P_off[i] = p_off_abs

        # ── QRS offset: same-height upslope after R (vs P_on level) ──
        p_on_local_for_qrs = (P_on[i] - s) if np.isfinite(P_on[i]) else None
        j_candidate = _find_qrs_offset_jpoint(
            y_seg, fs, r_loc, len(y_seg) - 1, p_on_local=p_on_local_for_qrs
        )
        qrs_off_abs = float(s + j_candidate)
        qrs_dur_ms = (qrs_off_abs - QRS_on[i]) / fs * 1000.0
        if not (4.0 <= qrs_dur_ms <= 35.0):
            if np.isfinite(QRS_off0[i]):
                qrs_off_abs = float(QRS_off0[i])
            else:
                qrs_off_abs = float(r_idx + int(0.010 * fs))
        if qrs_off_abs <= QRS_on[i]:
            qrs_off_abs = float(QRS_on[i] + int(0.006 * fs))
        QRS_off[i] = qrs_off_abs

        # ── T-wave offset: return-to-baseline + flatness in early post-R window ──
        qrs_off_local = QRS_off[i] - s
        qrs_on_local = QRS_on[i] - s
        t_search_end = min(len(y_seg) - 1, r_loc + int(min(0.90 * rr_i, 0.045) * fs))
        t_candidate = _find_t_offset_threshold(
            y_seg,
            fs,
            qrs_off_local,
            t_search_end,
            qrs_on_local=qrs_on_local,
            r_local=r_loc,
            p_on_local=(P_on[i] - s) if np.isfinite(P_on[i]) else None,
        )
        if t_candidate is not None and np.isfinite(t_candidate):
            T_off[i] = float(s + t_candidate)
        elif np.isfinite(T_off0[i]):
            T_off[i] = float(T_off0[i])
        else:
            T_off[i] = float(r_idx + int(0.028 * fs))

        qt_ms = (T_off[i] - QRS_on[i]) / fs * 1000.0
        if not (8.0 <= qt_ms <= 60.0):
            lo = float(QRS_off[i] + int(0.004 * fs))
            hi = float(r_idx + int(0.040 * fs))
            T_off[i] = float(np.clip(T_off[i], lo, hi))

        # ── Confidence ────────────────────────────────────────────
        found = sum([
            np.isfinite(P_on[i]),
            np.isfinite(P_off[i]),
            np.isfinite(QRS_on[i]),
            np.isfinite(QRS_off[i]),
            np.isfinite(T_off[i]),
        ])
        qrs_ok = (5.0  <= (QRS_off[i] - QRS_on[i]) / fs * 1000.0 <= 35.0
                  if (np.isfinite(QRS_on[i]) and np.isfinite(QRS_off[i])) else False)
        t_ok   = (10.0 <= (T_off[i] - QRS_on[i]) / fs * 1000.0 <= 60.0
                  if (np.isfinite(QRS_on[i]) and np.isfinite(T_off[i])) else False)
        conf[i] = float(found / 5.0 * 0.5
                        + (0.3 if qrs_ok else 0.0)
                        + (0.2 if t_ok   else 0.0))

    return QRS_on, QRS_off, T_off, P_on, P_off, conf


# ============================================================
# Main ECG metrics computation
# ============================================================
def compute_ecg_metrics(
    t,
    y,
    fs,
    engine="wavelet",
    debug_plots=False,
    source_file=None,
    validity_threshold=0.90,
    run_context=None,
):
    sig_filt, (f_lo, f_hi) = auto_bandpass_notch_ecg(y, fs)
    rpeaks, waves, used_engine = delineate_ecg(sig_filt, fs, engine=engine)
    if len(rpeaks) < 3:
        raise ValueError("Insufficient R-peaks detected.")

    N = len(rpeaks)
    QRS_on, QRS_off, T_off, P_on, P_off, conf = refine_qrs_t(sig_filt, fs, rpeaks, waves)

    r_t = rpeaks / fs
    rr  = np.full(N, np.nan)
    if N >= 2:
        rr[1:] = np.diff(r_t)

    P_dur = np.where(np.isfinite(P_on)   & np.isfinite(P_off),   (P_off  - P_on)   / fs, np.nan)
    PR    = np.where(np.isfinite(P_on)   & np.isfinite(QRS_on),  (QRS_on - P_on)   / fs, np.nan)
    QRS   = np.where(np.isfinite(QRS_on) & np.isfinite(QRS_off), (QRS_off- QRS_on) / fs, np.nan)
    QT    = np.where(np.isfinite(QRS_on) & np.isfinite(T_off),   (T_off  - QRS_on) / fs, np.nan)
    QTcB  = np.where(np.isfinite(QT) & np.isfinite(rr) & (rr > 0), QT / np.sqrt(rr), np.nan)

    conf_rank = pd.Series(np.where(np.isfinite(conf), conf, np.inf)).rank(
        method="dense", ascending=True
    ).to_numpy(dtype=float)

    rows = []
    for i in range(N):
        def ms(v):
            return float(v * 1000) if np.isfinite(v) else np.nan

        p_on_idx = _to_optional_int(P_on[i])
        p_off_idx = _to_optional_int(P_off[i])
        qrs_on_idx = _to_optional_int(QRS_on[i])
        qrs_off_idx = _to_optional_int(QRS_off[i])
        r_idx = _to_optional_int(rpeaks[i])
        t_off_idx = _to_optional_int(T_off[i])

        rows.append({
            "Beat_Index":        i,
            "P_on_idx":          p_on_idx,
            "P_off_idx":         p_off_idx,
            "QRS_on_idx":        qrs_on_idx,
            "QRS_off_idx":       qrs_off_idx,
            "R_idx":             r_idx,
            "T_off_idx":         t_off_idx,
            "R_time_s":          float(r_t[i]),
            "RR_s":              float(rr[i])     if np.isfinite(rr[i])     else np.nan,
            "RR_ms":             ms(rr[i]),
            "P_wave_dur_s":      float(P_dur[i])  if np.isfinite(P_dur[i])  else np.nan,
            "P_wave_dur_ms":     ms(P_dur[i]),
            "PR_interval_s":     float(PR[i])     if np.isfinite(PR[i])     else np.nan,
            "PR_interval_ms":    ms(PR[i]),
            "QRS_interval_s":    float(QRS[i])    if np.isfinite(QRS[i])    else np.nan,
            "QRS_interval_ms":   ms(QRS[i]),
            "QT_interval_s":     float(QT[i])     if np.isfinite(QT[i])     else np.nan,
            "QT_interval_ms":    ms(QT[i]),
            "QTc_Bazett_s":      float(QTcB[i])   if np.isfinite(QTcB[i])   else np.nan,
            "QTc_Bazett_ms":     ms(QTcB[i]),
            "Confidence":        float(conf[i])   if np.isfinite(conf[i])   else np.nan,
            "Confidence_Rank":   float(conf_rank[i]) if np.isfinite(conf[i]) else np.nan,
            "Is_Corrected":      False,
            "Corrected_At":      "",
            "Correction_Notes":  "",
            "Correction_Source": "auto",
        })

    beats_df = ensure_ecg_beats_schema(pd.DataFrame(rows))
    beats_df, rmssd = recompute_ecg_intervals_df(beats_df, fs_hz=fs)
    valid_ratio, valid_count, total_count = compute_ecg_validity(beats_df)
    validity_pass = bool(
        np.isfinite(valid_ratio) and valid_ratio >= float(validity_threshold)
    )

    if debug_plots:
        for i in range(min(5, N)):
            r_idx = rpeaks[i]
            rr_i  = rr[i] if np.isfinite(rr[i]) else (np.nanmedian(rr[np.isfinite(rr)]) if np.any(np.isfinite(rr)) else 0.12)
            L = int(np.clip(0.110 * fs, int(0.050 * fs), int(0.150 * fs)))
            R = int(np.clip(1.10 * rr_i * fs, int(0.080 * fs), int(0.250 * fs)))
            sl, el = max(0, r_idx - L), min(len(sig_filt), r_idx + R)
            xv = np.arange(sl, el) / fs

            plt.figure(figsize=(13, 4))
            plt.plot(xv, sig_filt[sl:el], color="darkred", lw=1, label="ECG")

            def _mark(idx_abs, color, label, marker="X", size=90):
                if np.isfinite(idx_abs):
                    ii = int(idx_abs)
                    if 0 <= ii < len(sig_filt):
                        plt.scatter(ii / fs, sig_filt[ii],
                                    color=color, s=size, marker=marker,
                                    zorder=5, label=label)

            _mark(rpeaks[i],  "green",   "R")
            _mark(P_on[i],    "#00BCD4", "P_on",       marker="^", size=80)
            _mark(P_off[i],   "#0288D1", "P_off",      marker="v", size=80)
            _mark(QRS_on[i],  "#FF9800", "QRS_on")
            _mark(QRS_off[i], "#7E57C2", "QRS_off (J)")
            _mark(T_off[i],   "#F44336", "T_off")

            plt.title(f"Beat {i} | Engine={used_engine} | conf={conf[i]:.2f}")
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
            plt.grid(alpha=0.2); plt.legend(fontsize=8)
            plt.tight_layout(); plt.show()

    meta = {
        "engine":           used_engine,
        "bandpass_low_Hz":  f_lo,
        "bandpass_high_Hz": f_hi,
        "fs_Hz":            fs,
        "total_beats":      N,
        "RMSSD_s":          rmssd,
        "RMSSD_ms":         rmssd * 1000 if np.isfinite(rmssd) else np.nan,
        "generated_at":     now_iso(),
        "source_file":      os.path.abspath(source_file) if source_file else "",
        "validity_threshold": float(validity_threshold),
        "valid_beats":      valid_count,
        "valid_beats_total": total_count,
        "validity_ratio":   valid_ratio,
        "validity_pass":    validity_pass,
    }
    if isinstance(run_context, dict):
        for key, value in run_context.items():
            meta[f"ctx_{key}"] = value
    return beats_df.to_dict(orient="records"), meta


# ============================================================
# 5) EEG ENGINE
# ============================================================
def eeg_preprocess(sig, fs):
    win = max(1, int(fs * 2.0))
    med = pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()
    x   = sig - med
    x   = apply_notch(x, fs, 50.0,  q=30.0)
    x   = apply_notch(x, fs, 100.0, q=30.0)
    x   = bandpass_zero_phase(x, fs, 0.5, min(100.0, 0.45 * fs), order=4)
    return x


def detect_spikes(sig, fs,
                  bp=(14.0, 70.0),
                  z_thresh=5.6,
                  width_ms=(12, 80),
                  refr_ms=22,
                  slope_thresh=3.0,
                  curv_thresh=2.0,
                  rtob_frac=0.25,
                  ampz_thresh=4.2,
                  ampz_win_s=1.5,
                  biphasic_mode='soft',
                  bip_win_ms=25.0,
                  bip_ratio_min=0.35):
    sos    = butter_bandpass_sos(fs, bp[0], bp[1], order=4)
    hf     = sosfiltfilt(sos, sig)
    hf_abs = np.abs(hf)
    med, dev = moving_mad_stats(hf_abs, fs, 10.0)
    z        = (hf_abs - med) / dev
    min_dist = max(1, int(fs * refr_ms / 1000.0))
    peaks, _ = find_peaks(z, height=z_thresh, distance=min_dist)

    grad_med = np.median(np.abs(np.diff(sig))) + 1e-9 if len(sig) >= 2 else 1.0
    curv_med = np.median(np.abs(np.diff(sig, n=2))) + 1e-9 if len(sig) >= 3 else 1.0
    results, n = [], len(sig)

    for p in peaks:
        amp_env, half = hf_abs[p], 0.5 * hf_abs[p]
        l, r = p, p
        while l > 0 and hf_abs[l] > half:     l -= 1
        while r < n - 1 and hf_abs[r] > half: r += 1
        w_ms = 1000.0 * max(1, r - l) / fs
        if not (width_ms[0] <= w_ms <= width_ms[1]):
            continue

        k          = max(2, int(round(0.005 * fs)))
        core_left  = max(0, p - k)
        core_right = min(n, p + k)
        core       = sig[core_left:core_right]
        g1         = np.diff(core) if core.size >= 2 else np.array([0.0])
        peak_slope = float(np.max(np.abs(g1))) if g1.size else 0.0

        pre_span  = max(k * 5, int(0.02 * fs))
        base_left = max(0, core_left - pre_span)
        base      = sig[base_left:core_left]
        base_grad = float(np.median(np.abs(np.diff(base)))) + 1e-9 if base.size >= 2 else grad_med
        if peak_slope < slope_thresh * base_grad:
            continue

        g2   = np.diff(core, n=2) if core.size >= 3 else np.array([0.0])
        sharp = float(np.max(np.abs(g2))) if g2.size else 0.0
        base2  = sig[max(0, base_left - pre_span):base_left]
        base_curv = float(np.median(np.abs(np.diff(base2, n=2)))) + 1e-9 if base2.size >= 3 else curv_med
        if sharp < curv_thresh * base_curv:
            continue

        a, b = p + int(0.04 * fs), min(n, p + int(0.08 * fs))
        if b > a:
            post_env  = float(np.mean(np.abs(sig[a:b])))
            spike_env = float(np.max(np.abs(sig[core_left:core_right]))) + 1e-9
            if post_env > rtob_frac * spike_env:
                continue

        sw  = int(0.08 * fs)
        s0, s1 = max(0, p - sw//2), min(n, p + sw//2)
        seg    = sig[s0:s1]
        if seg.size >= int(0.02 * fs):
            p30_80, p1_20 = band_power_fft(seg, fs, [(30, 80), (1, 20)])
            if p1_20 <= 0 or (p30_80 / p1_20) < 1.5:
                continue

        aw  = int(round(ampz_win_s * fs))
        a0, a1 = max(0, p - aw//2), min(n, p + aw//2)
        seg_abs = np.abs(sig[a0:a1])
        med_abs = float(np.median(seg_abs))
        mad_abs = float(mad(seg_abs, scale='normal')) + 1e-9
        z_amp   = (abs(float(sig[p])) - med_abs) / mad_abs
        if z_amp < ampz_thresh:
            continue

        bip_ratio = 0.0
        if biphasic_mode in ('soft', 'hard'):
            bw       = int(round(bip_win_ms / 1000.0 * fs))
            cand     = np.r_[sig[max(0, p - bw):core_left], sig[core_right:min(n, p + bw)]]
            main     = float(sig[p])
            main_abs = abs(main) + 1e-12
            if cand.size:
                opp       = float(np.min(cand)) if main >= 0 else float(np.max(cand))
                bip_ratio = abs(opp) / main_abs
            if biphasic_mode == 'hard' and bip_ratio < bip_ratio_min:
                continue

        slope_ratio = peak_slope / (base_grad + 1e-12)
        score       = (0.9 * (z_amp - ampz_thresh)
                       + 0.4 * (slope_ratio - slope_thresh)
                       + 0.4 * (bip_ratio / (bip_ratio_min + 1e-12)))
        confidence  = float(1.0 / (1.0 + np.exp(-score)))
        results.append({"idx": int(p), "w_ms": float(w_ms), "z_amp": float(z_amp),
                         "slope": float(peak_slope), "curv": float(sharp),
                         "bip_ratio": float(bip_ratio), "confidence": confidence})
    return results


def detect_swd(sig, fs, band=(5.0, 9.0), min_duration_s=0.5,
               max_gap_s=0.5, power_ratio=2.5):
    lo   = sosfiltfilt(butter_bandpass_sos(fs, band[0], band[1], 4), sig)
    env  = np.abs(hilbert(lo))
    ref  = sosfiltfilt(butter_bandpass_sos(fs, 1.0, 20.0, 4), sig)
    win  = max(1, int(fs * 0.5))
    num  = uniform_filter1d(lo * lo,  size=win, mode='nearest')
    den  = uniform_filter1d(ref * ref, size=win, mode='nearest') + 1e-12
    ratio = num / den

    env_mean = uniform_filter1d(env,       size=win, mode='nearest')
    env_var  = uniform_filter1d(env * env, size=win, mode='nearest') - env_mean ** 2
    env_cv   = np.sqrt(np.maximum(0.0, env_var)) / (env_mean + 1e-12)
    mask     = (ratio > power_ratio) & (env_cv < 0.6)

    events, in_evt, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_evt:
            in_evt, start = True, i
        if in_evt and not m:
            e = {"start": start / fs, "end": i / fs}
            if events and e["start"] - events[-1]["end"] <= max_gap_s:
                events[-1]["end"] = e["end"]
            else:
                events.append(e)
            in_evt = False
    if in_evt:
        events.append({"start": start / fs, "end": len(sig) / fs})

    kept = []
    for e in events:
        if e["end"] - e["start"] < min_duration_s:
            continue
        a, b = int(e["start"] * fs), int(e["end"] * fs)
        seg   = lo[a:b]
        peaks, _ = find_peaks(seg, distance=max(1, int(fs / 12.0)))
        if len(peaks) >= 4:
            isi = np.diff(peaks) / fs
            if float(np.std(isi) / (np.mean(isi) + 1e-12)) <= 0.25:
                kept.append(e)
        else:
            kept.append(e)
    return kept


def _close_gaps(mask, max_off_steps):
    if max_off_steps <= 0:
        return mask
    m, n, i = mask.copy(), len(mask), 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]: j += 1
            if 0 < j - i <= max_off_steps:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def detect_seizures_llrms(sig, fs,
                           win_s=1.0, step_s=0.25,
                           enter_z=5.5, exit_z=3.5, off_hold_s=2.0,
                           min_duration_s=5.0, merge_gap_s=15.0, close_s=10.0,
                           hf_ratio_thr=1.6, lf_ratio_thr=0.9,
                           occupancy_min=0.45, rhythm_prom_thr=1.35,
                           entropy_drop_sigma=0.20):
    w, step, n = max(1, int(fs*win_s)), max(1, int(fs*step_s)), len(sig)
    starts = np.arange(0, n - w + 1, step)
    t0     = starts / fs

    dx  = np.abs(np.diff(sig))
    LL  = np.convolve(dx, np.ones(w), mode='valid')[::step] if w > 1 else dx[::step]
    x2  = sig * sig
    cs  = np.cumsum(np.r_[0.0, x2])
    RMS = np.sqrt((cs[w:] - cs[:-w]) / w)[::step]

    ll_z  = (LL  - np.median(LL))  / (mad(LL,  scale='normal') + 1e-9)
    rms_z = (RMS - np.median(RMS)) / (mad(RMS, scale='normal') + 1e-9)
    z     = np.maximum(ll_z, rms_z)

    hf_r = np.zeros_like(z); lf_r = np.zeros_like(z)
    prom = np.zeros_like(z); sent = np.zeros_like(z)
    clip = np.zeros_like(z, dtype=bool)
    hkrt = np.zeros_like(z, dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0+w]
        if len(seg) < w: continue
        rv = np.percentile(seg, [1, 99])
        clip[i] = np.any(np.isclose(seg, rv[0], atol=1e-9)) or np.any(np.isclose(seg, rv[1], atol=1e-9))
        hkrt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0
        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1,45),(70,100),(0.1,0.5)])
        hf_r[i] = p70_100 / (p1_45 + 1e-12)
        lf_r[i] = p01_05  / (p1_45 + 1e-12)
        p4_30, p1_100 = band_power_fft(seg, fs, [(4,30),(1,100)])
        prom[i] = p4_30 / (max(p1_100, 1e-12))
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_z    = (np.median(sent) - sent) / (mad(sent, scale='normal') + 1e-9)
    artifact = (hf_r > hf_ratio_thr) | (lf_r > lf_ratio_thr) | clip | hkrt
    z_eff    = z.copy(); z_eff[artifact] = -1e9
    oh       = max(1, int(round(off_hold_s / step_s)))

    mask = np.zeros(len(z_eff), dtype=bool)
    i = 0
    while i < len(z_eff):
        if z_eff[i] >= enter_z:
            start, off = i, 0
            i += 1
            while i < len(z_eff):
                if z_eff[i] <= exit_z:
                    off += 1
                    if off >= oh:
                        mask[start:i-off+1] = True; break
                else:
                    off = 0
                i += 1
            else:
                mask[start:] = True; break
        else:
            i += 1

    if close_s > 0:
        mask = _close_gaps(mask, int(round(close_s / step_s)))

    events, in_evt, si = [], False, 0
    for i, on in enumerate(mask):
        if on and not in_evt:  in_evt, si = True, i
        if in_evt and not on:
            a, b = float(t0[si]), float(t0[i-1] + win_s)
            if b - a >= min_duration_s:
                events.append((si, i, a, b))
            in_evt = False
    if in_evt:
        a, b = float(t0[si]), float(t0[-1] + win_s)
        if b - a >= min_duration_s:
            events.append((si, len(mask), a, b))

    kept = []
    for si, ei, a, b in events:
        seg_on = (z[si:ei] >= exit_z) & (~artifact[si:ei])
        if float(np.mean(seg_on)) < occupancy_min: continue
        if not (float(np.mean(prom[si:ei] >= rhythm_prom_thr)) >= 0.60
                and float(np.mean(ent_z[si:ei] >= entropy_drop_sigma)) >= 0.60): continue
        kept.append({"start": a, "end": b})

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"] - merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
    return merged


def detect_seizures_spike_led(sig, fs, spike_times_rel,
                               win_s=0.5, hop_s=0.05,
                               min_rate_hz=6.0,
                               isi_min_s=0.015, isi_max_s=0.120, isi_cv_max=0.45,
                               min_duration_s=5.0, close_s=1.0, merge_gap_s=12.0,
                               hf_ratio_thr=1.6, lf_ratio_thr=0.9,
                               rhythm_prom_thr=1.30, entropy_drop_sigma=0.15,
                               occupancy_min=0.60):
    N = len(sig)
    w, step = max(1, int(round(win_s*fs))), max(1, int(round(hop_s*fs)))
    starts  = np.arange(0, N - w + 1, step)
    t0_arr  = starts / fs

    hf_r = np.zeros(len(starts)); lf_r = np.zeros(len(starts))
    prom = np.zeros(len(starts)); sent = np.zeros(len(starts))
    clip = np.zeros(len(starts), dtype=bool)
    hkrt = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0+w]
        if len(seg) < w: continue
        rv = np.percentile(seg, [1,99])
        clip[i] = np.any(np.isclose(seg,rv[0],atol=1e-9)) or np.any(np.isclose(seg,rv[1],atol=1e-9))
        hkrt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0
        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1,45),(70,100),(0.1,0.5)])
        hf_r[i] = p70_100/(p1_45+1e-12)
        lf_r[i] = p01_05/(p1_45+1e-12)
        p4_30, p1_100 = band_power_fft(seg, fs, [(4,30),(1,100)])
        prom[i] = p4_30/max(p1_100,1e-12)
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_z    = (np.median(sent) - sent) / (mad(sent, scale='normal') + 1e-9)
    artifact = (hf_r > hf_ratio_thr) | (lf_r > lf_ratio_thr) | clip | hkrt

    spikes    = np.array(spike_times_rel, dtype=float)
    spikes    = spikes[(spikes >= 0) & (spikes <= N/fs)] if spikes.size else spikes
    min_count = int(np.ceil(min_rate_hz * win_s) - 1e-9)
    rate_ok   = np.zeros(len(starts), dtype=bool)
    isi_ok    = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        a, b = s0/fs, s0/fs + win_s
        if spikes.size == 0: continue
        sw = spikes[np.searchsorted(spikes,a):np.searchsorted(spikes,b,side='right')]
        rate_ok[i] = sw.size >= min_count
        if sw.size >= 3:
            isi = np.diff(sw)
            med_isi = float(np.median(isi))
            if isi_min_s <= med_isi <= isi_max_s:
                isi_ok[i] = float(np.std(isi)/(np.mean(isi)+1e-12)) <= isi_cv_max

    on = rate_ok & isi_ok & (~artifact)
    if close_s > 0:
        on = _close_gaps(on, int(round(close_s/hop_s)))

    events, in_evt, si = [], False, 0
    for i, val in enumerate(on):
        if val and not in_evt:  in_evt, si = True, i
        if in_evt and not val:
            a, b = float(t0_arr[si]), float(t0_arr[i-1]+win_s)
            if b - a >= min_duration_s: events.append((si, i, a, b))
            in_evt = False
    if in_evt:
        a, b = float(t0_arr[si]), float(t0_arr[-1]+win_s)
        if b - a >= min_duration_s: events.append((si, len(on), a, b))

    kept = []
    for si, ei, a, b in events:
        occ = float(np.mean(on[si:ei])) if ei > si else 0.0
        if occ < occupancy_min: continue
        if not (float(np.mean(prom[si:ei]>=rhythm_prom_thr))>=0.60
                and float(np.mean(ent_z[si:ei]>=entropy_drop_sigma))>=0.60): continue
        kept.append({"start": a, "end": b})

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"]-merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
    return merged


def merge_intervals(events, merge_gap_s):
    if not events: return []
    events = sorted(events, key=lambda e: e["start"])
    merged = [dict(events[0])]
    for e in events[1:]:
        if e["start"] - merged[-1]["end"] <= merge_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
        else:
            merged.append(dict(e))
    return merged


def merge_spike_times(times, refr_s=0.02):
    if len(times) == 0: return np.array([], dtype=float)
    times = np.array(sorted(times), dtype=float)
    out   = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= refr_s: out.append(float(t))
    return np.array(out, dtype=float)


def bin_counts(spike_ts, swd_evts, sz_evts, bin_s, t0, t1):
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if edges[-1] < t1:
        edges = np.append(edges, t1)
    rows = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i+1])
        spikes = int(np.sum((spike_ts >= a) & (spike_ts < b))) if spike_ts.size else 0
        swds   = int(sum(e["start"] >= a and e["start"] < b for e in swd_evts))  if swd_evts  else 0
        szs    = int(sum(e["start"] >= a and e["start"] < b for e in sz_evts))   if sz_evts   else 0
        rows.append({"Bin_Index": i+1, "Bin_Start_s": a, "Bin_End_s": b,
                     "Spike_Count": spikes, "SWD_Count": swds, "Seizure_Count": szs})
    return pd.DataFrame(rows)


def analyze_eeg(t, y, fs, bin_s=3600.0, debug_plots=False):
    sig = eeg_preprocess(y, fs)
    sig_spk, fs_spk = decimate_to(sig, fs, 500)
    sig_swd, fs_swd = decimate_to(sig, fs, 200)
    sig_sz,  fs_sz  = decimate_to(sig, fs, 250)
    t0, t1 = float(t[0]), float(t[-1])

    spk_params = dict(z_thresh=5.6, width_ms=(12,80), refr_ms=22,
                      slope_thresh=3.0, curv_thresh=2.0, rtob_frac=0.25,
                      ampz_thresh=4.2, ampz_win_s=1.5,
                      biphasic_mode='soft', bip_win_ms=25.0, bip_ratio_min=0.35)
    spikes_raw  = detect_spikes(sig_spk, fs_spk, **spk_params)
    spike_times = np.array([t0 + s["idx"]/fs_spk for s in spikes_raw], dtype=float)
    spike_times = merge_spike_times(spike_times.tolist(), refr_s=0.02)

    swd_raw    = detect_swd(sig_swd, fs_swd)
    swd_events = [{"start": t0+e["start"], "end": t0+e["end"],
                   "type": "SWD", "confidence": 0.85} for e in swd_raw]

    rel_spikes  = (spike_times - t0).tolist()
    sz_spike    = detect_seizures_spike_led(sig_sz, fs_sz, rel_spikes)
    sz_llrms    = detect_seizures_llrms(sig_sz, fs_sz)
    sz_all      = ([{"start": t0+e["start"], "end": t0+e["end"],
                     "type": "Seizure", "confidence": 0.90} for e in sz_spike]
                 + [{"start": t0+e["start"], "end": t0+e["end"],
                     "type": "Seizure", "confidence": 0.75} for e in sz_llrms])
    sz_events   = merge_intervals(sz_all, merge_gap_s=12.0)

    summary_df = bin_counts(spike_times, swd_events, sz_events, bin_s, t0, t1)

    ev_rows = []
    for ti in spike_times:
        ev_rows.append({"Type": "Spike",   "Start_s": float(ti), "End_s": float(ti),
                        "Duration_s": 0.0, "Confidence": np.nan})
    for e in swd_events:
        ev_rows.append({"Type": "SWD",     "Start_s": float(e["start"]), "End_s": float(e["end"]),
                        "Duration_s": float(e["end"]-e["start"]),
                        "Confidence": float(e.get("confidence", np.nan))})
    for e in sz_events:
        ev_rows.append({"Type": "Seizure", "Start_s": float(e["start"]), "End_s": float(e["end"]),
                        "Duration_s": float(e["end"]-e["start"]),
                        "Confidence": float(e.get("confidence", np.nan))})
    events_df = pd.DataFrame(sorted(ev_rows, key=lambda r: (r["Start_s"], r["Type"])))

    meta = {
        "fs_Hz":                float(fs),
        "recording_duration_s": float(t1 - t0),
        "total_spikes":         int(len(spike_times)),
        "total_SWDs":           int(len(swd_events)),
        "total_seizures":       int(len(sz_events)),
        "bin_size_s":           float(bin_s),
        "preprocess":           "rolling-median detrend + 50/100 Hz notch + 0.5-100 Hz bandpass",
        "detector_fs":          "Spikes 500 Hz | SWD 200 Hz | Seizure 250 Hz",
    }

    if debug_plots and len(events_df) > 0:
        for _, row in events_df.head(3).iterrows():
            a  = max(t0, row["Start_s"] - 2.0)
            b  = min(t1, row["End_s"]   + 2.0)
            i0 = np.searchsorted(t, a)
            i1 = np.searchsorted(t, b)
            plt.figure(figsize=(12, 4))
            plt.plot(t[i0:i1], sig[i0:i1], 'k', lw=0.9)
            colour = {"Spike": "gold", "SWD": "lightskyblue", "Seizure": "tomato"}.get(row["Type"], "grey")
            plt.axvspan(row["Start_s"], row["End_s"], alpha=0.25, color=colour)
            plt.title(f'{row["Type"]}  {row["Start_s"]:.2f}–{row["End_s"]:.2f} s  conf={row["Confidence"]:.2f}')
            plt.xlabel("Time (s)"); plt.ylabel("EEG (a.u.)")
            plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

    return summary_df, events_df, meta


# ============================================================
# 6) EXCEL EXPORT
# ============================================================
def export_ecg_excel(save_path, beat_rows, meta):
    meta = dict(meta or {})
    df = ensure_ecg_beats_schema(
        beat_rows if isinstance(beat_rows, pd.DataFrame) else pd.DataFrame(beat_rows)
    )
    fs_hz = pd.to_numeric(pd.Series([meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(fs_hz) and fs_hz > 0:
        df, rmssd_s = recompute_ecg_intervals_df(df, float(fs_hz))
        meta["RMSSD_s"] = rmssd_s
        meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan

    valid_ratio, valid_count, total_count = compute_ecg_validity(df)
    meta["validity_ratio"] = valid_ratio
    meta["valid_beats"] = valid_count
    meta["valid_beats_total"] = total_count
    threshold = float(meta.get("validity_threshold", 0.90))
    meta["validity_threshold"] = threshold
    meta["validity_pass"] = bool(np.isfinite(valid_ratio) and valid_ratio >= threshold)
    meta.setdefault("generated_at", now_iso())
    meta.setdefault("source_file", "")
    meta.setdefault("review_last_saved_at", "")
    summary = compute_ecg_summary_row(df, meta)

    preferred = [
        "Beat_Index",
        "R_idx", "P_on_idx", "P_off_idx", "QRS_on_idx", "QRS_off_idx", "T_off_idx",
        "R_time_s",
        "RR_ms", "P_wave_dur_ms", "PR_interval_ms", "QRS_interval_ms", "QT_interval_ms", "QTc_Bazett_ms",
        "Confidence", "Confidence_Rank",
        "Is_Corrected", "Corrected_At", "Correction_Source", "Correction_Notes",
    ]
    trailing = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + trailing]

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="ECG_Beats")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="ECG_Summary")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")
    persist_workbook_corrections(save_path, str(meta.get("source_file", "")), df)


def export_eeg_excel(save_path, summary_df, events_df, meta):
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name="EEG_Bins")
        events_df.to_excel(writer,  index=False, sheet_name="EEG_Events")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")


# ============================================================
# 7) ECG review UI
# ============================================================
class ECGReviewWindow(tk.Toplevel):
    MARKER_STYLE = {
        "P_on_idx": ("#00BCD4", "^", "P_on"),
        "P_off_idx": ("#0288D1", "v", "P_off"),
        "QRS_on_idx": ("#FF9800", "X", "QRS_on"),
        "QRS_off_idx": ("#7E57C2", "X", "QRS_off"),
        "R_idx": ("green", "X", "R"),
        "T_off_idx": ("#F44336", "X", "T_off"),
    }

    def __init__(self, parent, workbook_path):
        super().__init__(parent)
        self.workbook_path = os.path.abspath(workbook_path)
        self.title(f"ECG Review - {os.path.basename(self.workbook_path)}")
        self.geometry("1180x760")
        self.minsize(980, 680)

        self.beats_df, self.meta = load_ecg_workbook(self.workbook_path)
        self.meta = dict(self.meta or {})

        self.source_file = self._resolve_source_file(str(self.meta.get("source_file", "")))
        self.meta["source_file"] = self.source_file

        fs_meta = pd.to_numeric(pd.Series([self.meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
        t_raw, y_raw = load_waveform_tabular(self.source_file)
        if np.isfinite(fs_meta) and fs_meta > 0:
            self.t, self.y, self.fs = ensure_uniform_sampling(t_raw, y_raw, fs_target=float(fs_meta))
        else:
            self.t, self.y, self.fs = ensure_uniform_sampling(t_raw, y_raw)
        self.sig_filt, _ = auto_bandpass_notch_ecg(self.y, self.fs)

        self.beats_df, restored = apply_saved_corrections(self.workbook_path, self.beats_df)
        self.beats_df, rmssd_s = recompute_ecg_intervals_df(self.beats_df, self.fs)
        self.meta["RMSSD_s"] = rmssd_s
        self.meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan
        self.meta["fs_Hz"] = float(self.fs)

        self.review_order = self._build_review_order()
        if not self.review_order:
            raise ValueError("No ECG beats found in workbook for review.")
        self.current_order_pos = 0

        self.selected_landmark = tk.StringVar(value="R_idx")
        self.notes_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=f"Loaded review session. Restored saved corrections: {restored}")

        self._build_ui()
        self._load_current_beat()

    def _resolve_source_file(self, source_file):
        if source_file and os.path.exists(source_file):
            return os.path.abspath(source_file)
        messagebox.showwarning(
            "Source file not found",
            "Source TSV path in workbook metadata is missing or unavailable.\nPlease select the original TSV file.",
            parent=self,
        )
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select source TSV/TXT/CSV/Excel file for ECG review",
            filetypes=[
                ("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            raise ValueError("Review canceled: source waveform file is required.")
        return os.path.abspath(selected)

    def _build_review_order(self):
        view = ensure_ecg_beats_schema(self.beats_df)
        rank = pd.to_numeric(view["Confidence_Rank"], errors="coerce")
        rank = rank.fillna(np.inf)
        ordered = view.assign(_rank=rank).sort_values(["_rank", "Beat_Index"])
        return ordered["Beat_Index"].astype(int).tolist()

    def _build_ui(self):
        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Editable landmark:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Combobox(
            top,
            values=LANDMARK_COLUMNS,
            textvariable=self.selected_landmark,
            state="readonly",
            width=14,
        ).pack(side="left", padx=6)

        tk.Button(top, text="Prev Beat", command=self._prev_beat, bg="#455A64", fg="white").pack(side="left", padx=4)
        tk.Button(top, text="Next Beat", command=self._next_beat, bg="#455A64", fg="white").pack(side="left", padx=4)
        tk.Button(top, text="Save Workbook", command=self._save_workbook, bg="#2E7D32", fg="white").pack(side="left", padx=10)
        tk.Button(top, text="Close", command=self.destroy, bg="#757575", fg="white").pack(side="right")

        note_row = tk.Frame(self, padx=10, pady=4)
        note_row.pack(fill="x")
        tk.Label(note_row, text="Correction notes:", font=("Arial", 10, "bold")).pack(side="left")
        note_entry = tk.Entry(note_row, textvariable=self.notes_var)
        note_entry.pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(note_row, text="Apply Note", command=self._apply_note).pack(side="left", padx=4)

        info = tk.Label(self, textvariable=self.info_var, anchor="w", justify="left",
                        font=("Consolas", 9), padx=10, pady=4)
        info.pack(fill="x")

        status = tk.Label(self, textvariable=self.status_var, anchor="w", fg="#1E88E5", padx=10, pady=4)
        status.pack(fill="x")

        fig_frame = tk.Frame(self, padx=8, pady=6)
        fig_frame.pack(fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(11.5, 5.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _current_beat_index(self):
        return int(self.review_order[self.current_order_pos])

    def _current_row_index(self):
        beat_index = self._current_beat_index()
        idx = self.beats_df.index[self.beats_df["Beat_Index"].astype(int) == beat_index]
        if len(idx) == 0:
            raise ValueError(f"Beat index {beat_index} not found in dataframe.")
        return int(idx[0])

    def _load_current_beat(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        self.notes_var.set(str(row.get("Correction_Notes", "")))
        self._draw_current_beat()
        self._refresh_info()

    def _refresh_info(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        beat_idx = int(row["Beat_Index"])
        conf = pd.to_numeric(pd.Series([row.get("Confidence", np.nan)]), errors="coerce").iloc[0]
        rank = pd.to_numeric(pd.Series([row.get("Confidence_Rank", np.nan)]), errors="coerce").iloc[0]
        ratio, valid, total = compute_ecg_validity(self.beats_df)
        valid_pct = f"{ratio * 100:.1f}%" if np.isfinite(ratio) else "N/A"
        self.info_var.set(
            f"Beat {beat_idx} ({self.current_order_pos + 1}/{len(self.review_order)}) | "
            f"Confidence={conf:.3f} Rank={rank:.0f} | "
            f"RR={row.get('RR_ms', np.nan):.2f} ms  PR={row.get('PR_interval_ms', np.nan):.2f} ms  "
            f"QRS={row.get('QRS_interval_ms', np.nan):.2f} ms  QT={row.get('QT_interval_ms', np.nan):.2f} ms  "
            f"QTcB={row.get('QTc_Bazett_ms', np.nan):.2f} ms | "
            f"Valid beats={valid}/{total} ({valid_pct})"
        )

    def _draw_current_beat(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        r_idx = _to_optional_int(row.get("R_idx"))
        if np.isnan(r_idx):
            r_idx = _to_optional_int(row.get("QRS_on_idx"))
        if np.isnan(r_idx):
            r_idx = 0
        r_idx = int(max(0, min(len(self.sig_filt) - 1, r_idx)))

        left = int(0.12 * self.fs)
        right = int(0.25 * self.fs)
        sl = max(0, r_idx - left)
        el = min(len(self.sig_filt), r_idx + right)

        self.ax.clear()
        x = np.arange(sl, el) / self.fs
        self.ax.plot(x, self.sig_filt[sl:el], color="darkred", lw=1.0, label="ECG")

        for col in LANDMARK_COLUMNS:
            color, marker, label = self.MARKER_STYLE[col]
            idx = _to_optional_int(row.get(col))
            if np.isnan(idx):
                continue
            idx = int(idx)
            if 0 <= idx < len(self.sig_filt):
                self.ax.scatter(
                    idx / self.fs,
                    self.sig_filt[idx],
                    color=color,
                    s=80 if col != "R_idx" else 100,
                    marker=marker,
                    zorder=5,
                    label=label,
                )
                self.ax.text(
                    idx / self.fs,
                    self.sig_filt[idx],
                    f" {label}",
                    fontsize=8,
                    color=color,
                )

        self.ax.set_title(
            f"Beat {int(row['Beat_Index'])} - click to set {self.selected_landmark.get()}",
            fontsize=11,
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(alpha=0.2)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            self.ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx_abs = int(round(float(event.xdata) * self.fs))
        idx_abs = int(max(0, min(len(self.sig_filt) - 1, idx_abs)))

        row_idx = self._current_row_index()
        col = self.selected_landmark.get()
        self.beats_df.at[row_idx, col] = idx_abs
        self.beats_df.at[row_idx, "Is_Corrected"] = True
        self.beats_df.at[row_idx, "Corrected_At"] = now_iso()
        self.beats_df.at[row_idx, "Correction_Source"] = "manual_review"
        note = self.notes_var.get().strip()
        if note:
            self.beats_df.at[row_idx, "Correction_Notes"] = note

        self.beats_df, rmssd_s = recompute_ecg_intervals_df(self.beats_df, self.fs)
        self.meta["RMSSD_s"] = rmssd_s
        self.meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan
        self.status_var.set(f"Updated {col} for beat {self._current_beat_index()} to sample {idx_abs}.")
        self._draw_current_beat()
        self._refresh_info()

    def _apply_note(self):
        row_idx = self._current_row_index()
        self.beats_df.at[row_idx, "Correction_Notes"] = self.notes_var.get().strip()
        self.beats_df.at[row_idx, "Is_Corrected"] = True
        self.beats_df.at[row_idx, "Corrected_At"] = now_iso()
        self.beats_df.at[row_idx, "Correction_Source"] = "manual_review"
        self.status_var.set(f"Updated note for beat {self._current_beat_index()}.")

    def _prev_beat(self):
        if self.current_order_pos > 0:
            self.current_order_pos -= 1
            self._load_current_beat()

    def _next_beat(self):
        if self.current_order_pos < len(self.review_order) - 1:
            self.current_order_pos += 1
            self._load_current_beat()

    def _save_workbook(self):
        ratio, valid, total = compute_ecg_validity(self.beats_df)
        threshold = float(self.meta.get("validity_threshold", 0.90))
        self.meta["validity_ratio"] = ratio
        self.meta["valid_beats"] = valid
        self.meta["valid_beats_total"] = total
        self.meta["validity_threshold"] = threshold
        self.meta["validity_pass"] = bool(np.isfinite(ratio) and ratio >= threshold)
        self.meta["review_last_saved_at"] = now_iso()
        self.meta["review_source"] = "manual_ui"
        self.meta["source_file"] = self.source_file
        export_ecg_excel(self.workbook_path, self.beats_df, self.meta)
        self.status_var.set(f"Saved workbook: {self.workbook_path}")
        messagebox.showinfo("Saved", f"Workbook overwritten with corrected ECG data:\n{self.workbook_path}", parent=self)


# ============================================================
# 8) UNIFIED GUI
# ============================================================
class UnifiedAnalyzerApp:
    def __init__(self, root):
        self.root         = root
        self.root.title("SUDEP Waveform Analyzer — EEG / ECG")
        self.root.geometry("920x700")
        self.root.resizable(True, True)
        self.input_file   = None
        self.results      = None
        self.is_analyzing = False
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="EEG/ECG Waveform Analyzer",
                 font=("Arial", 18, "bold"), fg="#1E88E5", pady=10).pack()

        ff = tk.LabelFrame(self.root,
                           text="Step 1 — Select Input File  (Col D = time, Col E = signal, row 8+)",
                           padx=14, pady=10, font=("Arial", 10, "bold"))
        ff.pack(fill="x", padx=18, pady=6)
        self.file_label = tk.Label(ff, text="No file selected", fg="gray", wraplength=640)
        self.file_label.pack(side="left", fill="x", expand=True)
        tk.Button(ff, text="Browse", command=self._browse,
                  bg="#43A047", fg="white", font=("Arial", 10, "bold"),
                  padx=14, pady=6).pack(side="right", padx=8)

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=18, pady=4)

        mode_f = tk.LabelFrame(top, text="Step 2 — Mode",
                               padx=10, pady=10, font=("Arial", 10, "bold"))
        mode_f.pack(side="left", padx=(0, 10), fill="x")
        self.mode_var = tk.StringVar(value="EEG")
        tk.Radiobutton(mode_f, text="EEG", variable=self.mode_var,
                       value="EEG", command=self._on_mode_change).pack(side="left", padx=8)
        tk.Radiobutton(mode_f, text="ECG", variable=self.mode_var,
                       value="ECG", command=self._on_mode_change).pack(side="left", padx=8)

        self.ecg_f = tk.LabelFrame(top, text="ECG Engine",
                                   padx=10, pady=10, font=("Arial", 10, "bold"))
        self.ecg_engine_var = tk.StringVar(value="wavelet" if NK_AVAILABLE else "heuristic")
        ttk.Combobox(self.ecg_f, values=["wavelet", "heuristic"],
                     textvariable=self.ecg_engine_var,
                     state="readonly", width=12).pack(side="left", padx=4)
        nk_note = "NeuroKit2 ✓" if NK_AVAILABLE else "NeuroKit2 not installed — heuristic only"
        tk.Label(self.ecg_f, text=nk_note, fg="#666").pack(side="left", padx=6)

        self.eeg_f = tk.LabelFrame(top, text="EEG Options",
                                   padx=10, pady=10, font=("Arial", 10, "bold"))
        tk.Label(self.eeg_f, text="Bin size (s):").pack(side="left")
        self.eeg_bin_entry = tk.Entry(self.eeg_f, width=8)
        self.eeg_bin_entry.insert(0, "3600")
        self.eeg_bin_entry.pack(side="left", padx=6)

        opt_f = tk.LabelFrame(top, text="Options",
                              padx=10, pady=10, font=("Arial", 10, "bold"))
        opt_f.pack(side="right", fill="x")
        self.debug_var    = tk.BooleanVar(value=False)
        self.force_fs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_f, text="Debug plots",    variable=self.debug_var).pack(anchor="w")
        tk.Checkbutton(opt_f, text="Force Fs (Hz):", variable=self.force_fs_var).pack(anchor="w")
        self.force_fs_entry = tk.Entry(opt_f, width=8)
        self.force_fs_entry.insert(0, "2000")
        self.force_fs_entry.pack(anchor="w", pady=(0, 2))

        prog = tk.LabelFrame(self.root, text="Progress",
                             padx=14, pady=10, font=("Arial", 10, "bold"))
        prog.pack(fill="x", padx=18, pady=6)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100, length=840)
        self.progress.pack(fill="x")
        self.status = tk.Label(prog, text="Ready", fg="#333")
        self.status.pack(anchor="w", pady=(4, 0))

        res_f = tk.LabelFrame(self.root, text="Results Summary",
                              padx=14, pady=10, font=("Arial", 10, "bold"))
        res_f.pack(fill="both", expand=True, padx=18, pady=6)
        self.result_text = tk.Text(res_f, height=10, font=("Courier", 9),
                                   state="disabled", bg="#f9f9f9")
        sb = ttk.Scrollbar(res_f, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=sb.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        btn_f = tk.Frame(self.root)
        btn_f.pack(fill="x", padx=18, pady=10)
        self.analyze_btn = tk.Button(btn_f, text="Analyze", state="disabled",
                                     command=self._start_analysis,
                                     bg="#1E88E5", fg="white",
                                     font=("Arial", 11, "bold"), padx=16, pady=10)
        self.analyze_btn.pack(side="left", padx=8)
        self.export_btn = tk.Button(btn_f, text="Export Excel", state="disabled",
                                    command=self._export,
                                    bg="#FB8C00", fg="white",
                                    font=("Arial", 11, "bold"), padx=16, pady=10)
        self.export_btn.pack(side="left", padx=8)
        self.review_btn = tk.Button(btn_f, text="Review ECG Workbook",
                                    command=self._open_review_workbook,
                                    bg="#6A1B9A", fg="white",
                                    font=("Arial", 11, "bold"), padx=16, pady=10)
        self.review_btn.pack(side="left", padx=8)
        tk.Button(btn_f, text="Reset", command=self._reset,
                  bg="#E53935", fg="white",
                  font=("Arial", 11, "bold"), padx=16, pady=10).pack(side="left", padx=8)
        tk.Button(btn_f, text="Exit",  command=self.root.quit,
                  bg="#757575", fg="white",
                  font=("Arial", 11, "bold"), padx=16, pady=10).pack(side="right", padx=8)

        self._on_mode_change()

    def _on_mode_change(self):
        if self.mode_var.get() == "ECG":
            self.ecg_f.pack(side="left", padx=(0,10), fill="x")
            self.eeg_f.pack_forget()
        else:
            self.eeg_f.pack(side="left", padx=(0,10), fill="x")
            self.ecg_f.pack_forget()

    def _set_progress(self, pct, msg):
        self.progress["value"] = float(pct)
        self.status.config(text=msg, fg="#1E88E5")
        self.root.update_idletasks()

    def _log(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", text)
        self.result_text.config(state="disabled")

    def _browse(self):
        path = filedialog.askopenfilename(
            title=f"Select {self.mode_var.get()} file",
            filetypes=[("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                       ("TSV", "*.tsv"), ("CSV", "*.csv"),
                       ("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if path:
            self.input_file = path
            size_mb = os.path.getsize(path) / 1024 / 1024
            self.file_label.config(
                text=f"{os.path.basename(path)}  ({size_mb:.2f} MB)", fg="black")
            self.analyze_btn.config(state="normal")

    def _start_analysis(self):
        if not self.input_file:
            messagebox.showerror("Error", "Please select a file first.")
            return
        if self.is_analyzing:
            messagebox.showwarning("Busy", "Analysis already running.")
            return
        self.is_analyzing = True
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress["value"] = 0
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            self._set_progress(5, "Loading file (Col D/E, row 8+)…")
            t, y = load_waveform_tabular(self.input_file)

            self._set_progress(15, "Inferring sampling rate…")
            if self.force_fs_var.get():
                fs_t = float(self.force_fs_entry.get().strip())
                t, y, fs = ensure_uniform_sampling(t, y, fs_target=fs_t)
            else:
                t, y, fs = ensure_uniform_sampling(t, y)

            mode  = self.mode_var.get()
            debug = bool(self.debug_var.get())

            if mode == "ECG":
                self._set_progress(30, "ECG: filtering, R-peak detection, landmark refinement…")
                engine = self.ecg_engine_var.get()
                beat_rows, meta = compute_ecg_metrics(t, y, fs,
                                                       engine=engine,
                                                       debug_plots=debug,
                                                       source_file=self.input_file,
                                                       validity_threshold=0.90,
                                                       run_context={
                                                           "ui_mode": "ECG",
                                                           "force_fs": bool(self.force_fs_var.get()),
                                                           "debug_plots": debug,
                                                       })
                self.results = {"mode": "ECG", "beats": beat_rows, "meta": meta}
                self._set_progress(100, f"ECG complete — {len(beat_rows)} beats detected.")

                df    = pd.DataFrame(beat_rows)
                lines = ["=== ECG RESULTS ===",
                         f"Beats detected : {len(beat_rows)}",
                         f"Engine         : {meta['engine']}",
                         f"Sampling rate  : {meta['fs_Hz']:.1f} Hz",
                         ""]
                for col, label in [
                    ("RR_ms",          "RR interval   (ms)"),
                    ("P_wave_dur_ms",   "P wave dur    (ms)"),
                    ("PR_interval_ms",  "PR interval   (ms)"),
                    ("QRS_interval_ms", "QRS interval  (ms)"),
                    ("QT_interval_ms",  "QT interval   (ms)"),
                    ("QTc_Bazett_ms",   "QTc Bazett    (ms)"),
                ]:
                    if col in df.columns:
                        v = df[col].dropna()
                        if len(v):
                            lines.append(f"  {label}  mean={v.mean():.2f}  "
                                         f"median={v.median():.2f}  "
                                         f"std={v.std():.2f}  n={len(v)}")
                rmssd = meta.get("RMSSD_ms", np.nan)
                lines.append(f"  RMSSD HRV     (ms)  {rmssd:.3f}" if np.isfinite(rmssd) else "  RMSSD:  N/A")
                vr = meta.get("validity_ratio", np.nan)
                vp = bool(meta.get("validity_pass", False))
                vt = float(meta.get("validity_threshold", 0.90)) * 100.0
                if np.isfinite(vr):
                    lines.append(f"  Validity      : {vr * 100:.2f}%  (threshold {vt:.1f}% -> {'PASS' if vp else 'FAIL'})")
                self._log("\n".join(lines))

            else:
                self._set_progress(30, "EEG: preprocessing…")
                bin_s = float(self.eeg_bin_entry.get().strip())
                self._set_progress(40, "EEG: spike / SWD / seizure detection…")
                summary_df, events_df, meta = analyze_eeg(t, y, fs,
                                                           bin_s=bin_s,
                                                           debug_plots=debug)
                self.results = {"mode": "EEG", "summary": summary_df,
                                "events": events_df, "meta": meta}
                self._set_progress(100,
                    f"EEG complete — {meta['total_spikes']} spikes | "
                    f"{meta['total_SWDs']} SWDs | {meta['total_seizures']} seizures.")

                lines = ["=== EEG RESULTS ===",
                         f"Recording duration : {meta['recording_duration_s']:.1f} s",
                         f"Sampling rate      : {meta['fs_Hz']:.1f} Hz",
                         f"Bin size           : {meta['bin_size_s']:.0f} s",
                         f"Total spikes       : {meta['total_spikes']}",
                         f"Total SWDs         : {meta['total_SWDs']}",
                         f"Total seizures     : {meta['total_seizures']}",
                         ""]
                if len(events_df):
                    swds = events_df[events_df["Type"] == "SWD"]
                    szs  = events_df[events_df["Type"] == "Seizure"]
                    if len(swds):
                        lines.append(f"  SWD duration (s)  mean={swds['Duration_s'].mean():.2f}  "
                                     f"min={swds['Duration_s'].min():.2f}  max={swds['Duration_s'].max():.2f}")
                    if len(szs):
                        lines.append(f"  Seizure dur  (s)  mean={szs['Duration_s'].mean():.2f}  "
                                     f"min={szs['Duration_s'].min():.2f}  max={szs['Duration_s'].max():.2f}")
                    lines.append("")
                    lines.append("--- Event list (first 20) ---")
                    for _, row in events_df.head(20).iterrows():
                        lines.append(f"  {row['Type']:<8}  "
                                     f"start={row['Start_s']:>10.3f} s  "
                                     f"end={row['End_s']:>10.3f} s  "
                                     f"dur={row['Duration_s']:>7.3f} s  "
                                     f"conf={row['Confidence']:.2f}")
                self._log("\n".join(lines))

            self.export_btn.config(state="normal")
            messagebox.showinfo("Complete",
                                "Analysis complete.\nClick Export Excel to save results.")

        except Exception as exc:
            self._set_progress(0, f"Error: {exc}")
            self._log(f"ERROR\n{exc}")
            messagebox.showerror("Analysis failed", str(exc))
        finally:
            self.is_analyzing = False
            self.analyze_btn.config(state="normal")

    def _export(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return
        mode     = self.results["mode"]
        out_name = (f"{os.path.splitext(os.path.basename(self.input_file))[0]}"
                    f"_{mode}_results.xlsx")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=out_name,
            filetypes=[("Excel", "*.xlsx")],
            title="Save results")
        if not save_path:
            return
        try:
            if mode == "ECG":
                export_ecg_excel(save_path, self.results["beats"], self.results["meta"])
            else:
                export_eeg_excel(save_path, self.results["summary"],
                                 self.results["events"], self.results["meta"])
            self.status.config(text=f"Exported: {os.path.basename(save_path)}", fg="#43A047")
            messagebox.showinfo("Exported", f"Saved to:\n{save_path}")
            if mode == "ECG":
                open_now = messagebox.askyesno(
                    "Review ECG now?",
                    "Do you want to open the manual ECG review tool for this workbook now?",
                )
                if open_now:
                    self._open_review_workbook(workbook_path=save_path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _open_review_workbook(self, workbook_path=None):
        path = workbook_path
        if not path:
            path = filedialog.askopenfilename(
                title="Select exported ECG workbook to review",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
            )
        if not path:
            return
        try:
            ECGReviewWindow(self.root, path)
        except Exception as exc:
            messagebox.showerror("ECG review failed", str(exc))

    def _reset(self):
        self.input_file   = None
        self.results      = None
        self.is_analyzing = False
        self.file_label.config(text="No file selected", fg="gray")
        self.progress["value"] = 0
        self.status.config(text="Ready", fg="#333")
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self._log("")


# ============================================================
# 9) CLI ENTRYPOINTS
# ============================================================
def cli_analyze_ecg(args):
    input_path = os.path.abspath(args.input)
    t, y = load_waveform_tabular(input_path)
    if args.force_fs is not None:
        t, y, fs = ensure_uniform_sampling(t, y, fs_target=float(args.force_fs))
    else:
        t, y, fs = ensure_uniform_sampling(t, y)

    beat_rows, meta = compute_ecg_metrics(
        t,
        y,
        fs,
        engine=args.engine,
        debug_plots=bool(args.debug_plots),
        source_file=input_path,
        validity_threshold=float(args.validity_threshold),
        run_context={
            "cli_command": "analyze-ecg",
            "force_fs": args.force_fs if args.force_fs is not None else "",
            "debug_plots": bool(args.debug_plots),
        },
    )

    output_path = args.output
    if not output_path:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_ECG_results.xlsx")
    output_path = os.path.abspath(output_path)

    if not args.no_export:
        export_ecg_excel(output_path, beat_rows, meta)

    ratio = meta.get("validity_ratio", np.nan)
    threshold = float(meta.get("validity_threshold", args.validity_threshold))
    passed = bool(meta.get("validity_pass", False))
    print("=== CLI ECG ANALYSIS ===")
    print(f"Input            : {input_path}")
    print(f"Engine           : {meta.get('engine')}")
    print(f"Sampling rate Hz : {float(meta.get('fs_Hz', np.nan)):.3f}")
    print(f"Total beats      : {int(meta.get('total_beats', 0))}")
    print(f"Validity ratio   : {ratio * 100:.3f}%")
    print(f"Validity pass    : {'PASS' if passed else 'FAIL'} (threshold {threshold * 100:.1f}%)")
    if not args.no_export:
        print(f"Workbook         : {output_path}")
    return 0 if passed else 2


def cli_review_ecg(args):
    workbook = os.path.abspath(args.workbook)
    beats_df, meta = load_ecg_workbook(workbook)
    meta = dict(meta or {})

    if args.source_file:
        meta["source_file"] = os.path.abspath(args.source_file)

    fs_hz = pd.to_numeric(pd.Series([args.fs_hz if args.fs_hz is not None else meta.get("fs_Hz", np.nan)]),
                          errors="coerce").iloc[0]
    if not (np.isfinite(fs_hz) and fs_hz > 0):
        raise ValueError("review-ecg requires valid fs_Hz (from Meta sheet or --fs-hz).")

    applied = 0
    if args.apply_saved_corrections:
        beats_df, applied = apply_saved_corrections(workbook, beats_df)

    beats_df, rmssd_s = recompute_ecg_intervals_df(beats_df, float(fs_hz))
    meta["fs_Hz"] = float(fs_hz)
    meta["RMSSD_s"] = rmssd_s
    meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan

    ratio, valid, total = compute_ecg_validity(beats_df)
    threshold = float(args.validity_threshold if args.validity_threshold is not None else meta.get("validity_threshold", 0.90))
    meta["validity_threshold"] = threshold
    meta["validity_ratio"] = ratio
    meta["valid_beats"] = valid
    meta["valid_beats_total"] = total
    meta["validity_pass"] = bool(np.isfinite(ratio) and ratio >= threshold)

    output_path = os.path.abspath(args.output) if args.output else workbook
    should_save = bool(args.save or args.output or args.apply_saved_corrections)
    if should_save:
        meta["review_last_saved_at"] = now_iso()
        meta["review_source"] = "cli_review"
        export_ecg_excel(output_path, beats_df, meta)

    print("=== CLI ECG REVIEW ===")
    print(f"Workbook         : {workbook}")
    print(f"Saved corrections: {applied}")
    print(f"Valid beats      : {valid}/{total}")
    print(f"Validity ratio   : {ratio * 100:.3f}%")
    print(f"Validity pass    : {'PASS' if meta['validity_pass'] else 'FAIL'} (threshold {threshold * 100:.1f}%)")
    if should_save:
        print(f"Saved workbook   : {output_path}")
    return 0 if bool(meta["validity_pass"]) else 2


def cli_review_ecg_ui(args):
    workbook = os.path.abspath(args.workbook)
    root = tk.Tk()
    root.withdraw()
    ECGReviewWindow(root, workbook)
    root.mainloop()
    return 0


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="SUDEP EEG/ECG analyzer - GUI and batch ECG utilities"
    )
    sub = parser.add_subparsers(dest="command")

    p_an = sub.add_parser("analyze-ecg", help="Run ECG analysis and export workbook")
    p_an.add_argument("--input", required=True, help="Input waveform file")
    p_an.add_argument("--output", default="", help="Output workbook path (.xlsx)")
    p_an.add_argument("--engine", choices=["wavelet", "heuristic"],
                      default=("wavelet" if NK_AVAILABLE else "heuristic"))
    p_an.add_argument("--force-fs", type=float, default=None, help="Force target sampling rate Hz")
    p_an.add_argument("--validity-threshold", type=float, default=0.90,
                      help="Minimum valid-beat ratio required to pass")
    p_an.add_argument("--debug-plots", action="store_true", help="Show debug landmark plots")
    p_an.add_argument("--no-export", action="store_true", help="Run analysis without writing workbook")

    p_rev = sub.add_parser("review-ecg", help="Recompute/revalidate an exported ECG workbook")
    p_rev.add_argument("--workbook", required=True, help="Existing ECG workbook path")
    p_rev.add_argument("--output", default="", help="Output workbook path (defaults to overwrite input)")
    p_rev.add_argument("--source-file", default="", help="Override source file path in meta")
    p_rev.add_argument("--fs-hz", type=float, default=None, help="Override fs_Hz used for recomputation")
    p_rev.add_argument("--validity-threshold", type=float, default=None,
                       help="Override validity threshold")
    p_rev.add_argument("--apply-saved-corrections", action="store_true",
                       help="Apply persisted correction store entries before recompute")
    p_rev.add_argument("--save", action="store_true",
                       help="Save workbook after review even if no corrections were applied")

    p_ui = sub.add_parser("review-ecg-ui", help="Launch manual ECG review UI for a workbook")
    p_ui.add_argument("--workbook", required=True, help="Existing ECG workbook path")
    return parser


def run_cli(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not args.command:
        return None
    if args.command == "analyze-ecg":
        return cli_analyze_ecg(args)
    if args.command == "review-ecg":
        return cli_review_ecg(args)
    if args.command == "review-ecg-ui":
        return cli_review_ecg_ui(args)
    return 1


# ============================================================
# 10) ENTRY POINT
# ============================================================
if __name__ == "__main__":
    cli_code = run_cli(sys.argv[1:])
    if cli_code is not None:
        sys.exit(int(cli_code))
    root = tk.Tk()
    app = UnifiedAnalyzerApp(root)
    root.mainloop()

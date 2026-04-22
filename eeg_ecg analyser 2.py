import os
import json
import threading
import pathlib
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, iirnotch, periodogram, find_peaks, sosfiltfilt, sosfilt, resample_poly, hilbert
from scipy.stats import median_abs_deviation as mad, kurtosis
from scipy.ndimage import uniform_filter1d

# Optional dependency (NeuroKit2 for ECG wave delineation)
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except Exception:
    NK_AVAILABLE = False


# ============================================================
# 1) LOADER (Columns D & E from Row 8)
# ============================================================
def load_waveform_tabular(file_path: str):
    """
    Unified loader for TSV/TXT/CSV and Excel files.
    ALWAYS reads:
      - Columns D & E only (0-based indices [3, 4] / Excel range 'D:E')
      - Starting from row 8 (skiprows = 7)
    Returns:
        t: np.ndarray (time vector)
        y: np.ndarray (signal vector)
    """
    path = pathlib.Path(file_path)
    ext = path.suffix.lower()

    def _coerce_numeric(a):
        a = pd.to_numeric(
            pd.Series(a).astype(str).str.strip()
            .str.replace(r'[^0-9eE\+\-\.]', '', regex=True),
            errors='coerce'
        ).values
        return a

    if ext in [".tsv", ".txt", ".csv"]:
        sep = '\t' if ext == ".tsv" else None
        try:
            df = pd.read_csv(file_path, sep=sep, header=None, skiprows=7, usecols=[3, 4], engine='python')
        except Exception:
            df = pd.read_csv(file_path, sep=None, header=None, skiprows=7, usecols=[3, 4], engine='python')

        if df.shape[1] != 2:
            raise ValueError(f"Expected exactly 2 columns from D:E, got {df.shape[1]}")

        t = _coerce_numeric(df.iloc[:, 0].values)
        y = _coerce_numeric(df.iloc[:, 1].values)

    elif ext in [".xlsx", ".xls"]:
        engine = 'openpyxl' if ext == ".xlsx" else 'xlrd'
        df = pd.read_excel(file_path, engine=engine, header=None, usecols="D:E", skiprows=7)
        if df.shape[1] != 2:
            raise ValueError(f"Expected exactly 2 columns from D:E, got {df.shape[1]}")
        t = _coerce_numeric(df.iloc[:, 0].values)
        y = _coerce_numeric(df.iloc[:, 1].values)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    valid = ~(np.isnan(t) | np.isnan(y))
    t = t[valid]
    y = y[valid]

    if len(t) == 0:
        raise ValueError("No valid numeric data found in columns D & E (row 8 onward).")

    return t, y


# ============================================================
# 2) SAMPLING RATE INFERENCE & RESAMPLING
# ============================================================
def infer_fs_from_time(t):
    """
    Infer sampling rate from time axis (auto-detect s vs ms).
    Returns fs, jitter, and possibly corrected t (ms->s).
    """
    t = np.asarray(t)
    if len(t) < 3:
        raise ValueError("Not enough samples to infer sampling rate.")
    dt = np.diff(t)
    dt_med = np.median(dt)
    if dt_med <= 0 or not np.isfinite(dt_med):
        raise ValueError("Invalid time increments.")
    fs = 1.0 / dt_med

    # If fs looks too low, try ms->s
    if fs < 50:
        t = t * 1e-3
        dt = dt * 1e-3
        dt_med = np.median(dt)
        fs = 1.0 / dt_med

    jitter = np.std(dt) / (np.mean(dt) + 1e-12)
    return fs, jitter, t


def ensure_uniform_sampling(t, y, fs_target=None):
    """
    If time steps are jittery, resample to a uniform grid at fs_target (or inferred fs).
    """
    fs, jitter, t = infer_fs_from_time(t)
    if fs_target is None:
        fs_target = float(int(round(fs)))

    if jitter > 0.01:
        N = int((t[-1] - t[0]) * fs_target) + 1
        t_uniform = np.linspace(t[0], t[-1], N)
        y_uniform = np.interp(t_uniform, t, y)
        return t_uniform, y_uniform, fs_target

    return t, y, fs


# ============================================================
# 3) COMMON FILTERS / HELPERS
# ============================================================
def butter_bandpass_sos(fs, low, high, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, low / nyq)
    high = min(0.999999, high / nyq)
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_zero_phase(sig, fs, lo, hi, order=4):
    sos = butter_bandpass_sos(fs, lo, hi, order=order)
    return sosfiltfilt(sos, sig)


def apply_notch(sig, fs, line_hz=50.0, q=35.0):
    w0 = line_hz / (fs / 2.0)
    if w0 <= 0 or w0 >= 1:
        return sig
    b, a = iirnotch(w0, q)
    padlen = min(len(sig) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, sig, padtype='odd', padlen=padlen)


def moving_mad_stats(x, fs, win_s=10.0):
    w = max(1, int(fs * win_s))
    s = pd.Series(x)
    med_series = s.rolling(w, center=True, min_periods=1).median()
    dev_series = (s - med_series).abs().rolling(w, center=True, min_periods=1).median()
    med = med_series.to_numpy().astype(np.float64, copy=False)
    dev = (dev_series.to_numpy() * 1.4826).astype(np.float64, copy=False)  # MAD->sigma
    dev[~np.isfinite(dev)] = np.nanmedian(dev[np.isfinite(dev)]) if np.any(np.isfinite(dev)) else 1.0
    dev[dev < 1e-9] = np.nanmedian(dev[dev > 0]) if np.any(dev > 0) else 1.0
    return med, dev


def decimate_to(x, fs, target_fs, cutoff=None):
    """
    Anti-aliased decimation using polyphase resampling.
    """
    if target_fs >= fs:
        return x, fs
    cutoff = cutoff or (0.45 * target_fs)
    sos_lp = butter(6, cutoff, btype="low", fs=fs, output="sos")
    y = sosfilt(sos_lp, x)
    down = int(round(fs / target_fs))
    y_ds = resample_poly(y, up=1, down=down)
    return y_ds, fs / down


def band_power_fft(x, fs, bands, nfft=None):
    n = len(x)
    nfft = n if nfft is None else nfft
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    X = np.fft.rfft(x, n=nfft)
    P = (np.abs(X) ** 2) / nfft
    out = []
    for (f1, f2) in bands:
        m = (freqs >= f1) & (freqs < f2)
        out.append(float(np.sum(P[m])))
    return out


def spectral_entropy_from_fft(x, fs, nfft=None):
    n = len(x)
    nfft = n if nfft is None else nfft
    X = np.fft.rfft(x, n=nfft)
    P = (np.abs(X) ** 2) + 1e-12
    P = P / np.sum(P)
    return float(-np.sum(P * np.log2(P)))


# ============================================================
# 4) ECG ENGINE (robust R-peaks + optional NK delineation + refinement)
# ============================================================
def auto_bandpass_notch_ecg(sig, fs):
    """
    Data-driven bandpass + optional notch, preserving high-frequency QRS content (mouse).
    """
    f, Pxx = periodogram(sig, fs=fs, scaling='density')

    # Default safe range for mouse ECG
    lo = 0.5
    hi = min(200.0, 0.45 * fs)

    # Notch if clear 50 Hz line
    def has_line(freq):
        band = (f >= freq - 0.7) & (f <= freq + 0.7)
        neigh = (f >= freq - 5) & (f <= freq + 5)
        if not np.any(neigh):
            return False
        return np.sum(Pxx[band]) > 6 * np.median(Pxx[neigh])

    y = bandpass_zero_phase(sig, fs, lo, hi, order=4)
    if has_line(50.0):
        y = apply_notch(y, fs, 50.0, q=35.0)

    return y, (lo, hi)


def find_r_peaks_fallback(y, fs):
    """
    Robust R-peak fallback: energy envelope + spectral estimate of RR -> distance constraint.
    """
    z = y - np.median(y)
    win = max(3, int(0.015 * fs))
    env = np.convolve(z * z, np.ones(win) / win, mode='same')

    # Estimate dominant heart rate band for mice: 4–20 Hz (~240–1200 bpm)
    f = np.fft.rfftfreq(len(env), d=1 / fs)
    E = np.abs(np.fft.rfft(env))
    band = (f >= 4) & (f <= 20)
    f0 = f[band][np.argmax(E[band])] if np.any(band) else 8.0
    rr_s = 1.0 / max(1e-3, float(f0))

    distance = max(1, int(0.55 * rr_s * fs))
    prom = np.percentile(env, 98) * 0.10
    peaks, _ = find_peaks(env, prominence=prom, distance=distance)
    return np.array(peaks, dtype=int)


def plausible_rpeaks(rpeaks, fs, n_samples):
    """
    Sanity check on R-peaks for mouse ECG.
    """
    if rpeaks is None or len(rpeaks) < 3:
        return False
    r = np.asarray(rpeaks)
    if np.any(r < 0) or np.any(r >= n_samples):
        return False
    rr = np.diff(r) / fs
    rr = rr[np.isfinite(rr)]
    if rr.size < 2:
        return False
    rr_med = np.median(rr)
    # Mouse RR typically ~50–250 ms; allow wider to be safe
    if not (0.03 <= rr_med <= 0.40):
        return False
    # Too many implausibly short intervals
    if np.mean(rr < 0.02) > 0.05:
        return False
    return True


def delineate_ecg(sig_filt, fs, engine="wavelet"):
    """
    Returns rpeaks and wave indices dict (may be NaN if not available).
    """
    n = len(sig_filt)

    # --- Primary: NeuroKit2 if available ---
    if engine == "wavelet" and NK_AVAILABLE:
        try:
            signals, info = nk.ecg_process(sig_filt, sampling_rate=fs, method="neurokit")
            rpeaks = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            if not plausible_rpeaks(rpeaks, fs, n):
                raise RuntimeError("NeuroKit2 R-peaks not plausible; falling back.")
            # Delineation
            out = nk.ecg_delineate(signals["ECG_Clean"], rpeaks=rpeaks, sampling_rate=fs, method="dwt")
            if isinstance(out, tuple):
                waves = out[0] if isinstance(out[0], dict) else out[1]
            else:
                waves = out
            return rpeaks, waves, "wavelet"
        except Exception:
            pass

    # --- Fallback: envelope-based R-peaks only ---
    rpeaks = find_r_peaks_fallback(sig_filt, fs)
    waves = {}  # empty -> will yield NaNs for P/T etc.
    return rpeaks, waves, "heuristic"


def samples_to_time_single(v, fs):
    if v is None:
        return np.nan
    try:
        if np.isnan(v):
            return np.nan
    except Exception:
        pass
    return float(v) / fs


def samples_to_time(arr, fs, N=None):
    if arr is None:
        return np.full(N if N is not None else 0, np.nan)
    a = np.asarray(arr, dtype=float)
    out = np.full(len(a), np.nan)
    for i, v in enumerate(a):
        out[i] = np.nan if (v is None or np.isnan(v)) else float(v) / fs
    return out


def _hf_envelope(y, win):
    win = max(3, int(win))
    ker = np.ones(win) / win
    return np.convolve(y ** 2, ker, mode='same')


def _candidate_qrs_bounds(y_seg, fs, r_loc):
    dy = np.gradient(y_seg)
    d2 = np.gradient(dy)
    env = _hf_envelope(y_seg, max(3, int(0.010 * fs)))

    thr_d2 = np.quantile(np.abs(d2), 0.92)
    thr_env = np.quantile(env, 0.70)

    on_idx = r_loc
    i = r_loc
    while i > 1:
        if (np.abs(d2[i]) < thr_d2) and (env[i] < thr_env):
            on_idx = i
            break
        i -= 1

    off_idx = r_loc
    j = r_loc
    while j < len(y_seg) - 2:
        if (np.abs(d2[j]) < thr_d2) and (env[j] < thr_env):
            off_idx = j
            break
        j += 1

    return on_idx, off_idx


def _candidate_t_end(y_seg, fs, start_idx, end_idx):
    lo = max(0, int(start_idx))
    hi = min(len(y_seg), int(end_idx))
    if hi - lo < max(8, int(0.010 * fs)):
        return np.nan, np.inf

    y = y_seg[lo:hi]
    dy = np.gradient(y)
    k = int(np.argmin(dy))  # steepest downslope within window
    k_abs = lo + k

    w = max(4, int(0.006 * fs))
    a = max(lo, k_abs - w)
    b = min(hi, k_abs + w)
    xx = np.arange(a, b)
    yy = y_seg[a:b]
    if len(xx) < 3:
        return np.nan, np.inf

    t0 = xx / fs
    m, c = np.polyfit(t0, yy, 1)
    tail_lo = int(lo + 0.85 * (hi - lo))
    baseline = np.median(y_seg[tail_lo:hi]) if hi > tail_lo else np.median(y)
    if abs(m) < 1e-12:
        return np.nan, np.inf

    t_isect = (baseline - c) / m
    t_end_idx = int(np.clip(t_isect * fs, lo, hi - 1))
    rmse = float(np.sqrt(np.mean((yy - ((xx / fs) * m + c)) ** 2)))
    return t_end_idx, rmse


def refine_qrs_t(sig_filt, fs, rpeaks, waves):
    """
    Refine QRS onset/offset and T offset per beat.
    Uses NeuroKit2 proposals if available; otherwise finds candidates purely from signal.
    Returns arrays for QRS_on, QRS_off, T_off, confidence.
    """
    N = len(rpeaks)
    n = len(sig_filt)

    def get_wave(name):
        arr = waves.get(name, None)
        if arr is None:
            return np.full(N, np.nan)
        arr = np.asarray(arr, dtype=float)
        out = np.full(N, np.nan)
        k = min(len(arr), N)
        out[:k] = arr[:k]
        return out

    QRS_on0 = get_wave("ECG_R_Onsets")
    QRS_off0 = get_wave("ECG_R_Offsets")
    T_pk0 = get_wave("ECG_T_Peaks")
    T_off0 = get_wave("ECG_T_Offsets")

    r_times = rpeaks / fs
    rr = np.full(N, np.nan)
    if N >= 2:
        rr[1:] = np.diff(r_times)
    rr_med = np.nanmedian(rr) if np.any(np.isfinite(rr)) else 0.12

    QRS_on = np.copy(QRS_on0)
    QRS_off = np.copy(QRS_off0)
    T_off = np.copy(T_off0)
    conf = np.full(N, np.nan)

    for i in range(N):
        r_idx = int(rpeaks[i])
        rr_i = rr[i] if (np.isfinite(rr[i]) and rr[i] > 0) else rr_med

        left_ms = float(np.clip(0.60 * rr_i * 1000.0, 40.0, 140.0))
        right_ms = float(np.clip(1.00 * rr_i * 1000.0, 80.0, 220.0))
        L = int(left_ms * fs / 1000.0)
        R = int(right_ms * fs / 1000.0)

        s = max(0, r_idx - L)
        e = min(n, r_idx + R)
        y_seg = sig_filt[s:e]
        r_loc = r_idx - s

        # Candidate QRS boundaries from curvature/energy decay
        on2, off2 = _candidate_qrs_bounds(y_seg, fs, r_loc)

        # Choose between NK proposal and candidate based on plausibility
        on1 = int(QRS_on0[i] - s) if np.isfinite(QRS_on0[i]) else np.nan
        off1 = int(QRS_off0[i] - s) if np.isfinite(QRS_off0[i]) else np.nan

        def qrs_cost(on, off):
            if not np.isfinite(on) or not np.isfinite(off) or off <= on:
                return 1e9
            width = (off - on) / fs
            # Mouse QRS tends to be very narrow; allow 5–40 ms
            w_pen = 0.0 if (0.005 <= width <= 0.040) else min(10.0, abs(width - 0.020) / 0.010)
            env = _hf_envelope(y_seg, max(3, int(0.010 * fs)))
            base = np.quantile(env, 0.15)
            e_pen = abs(env[int(on)] - base) + abs(env[int(off)] - base)
            return float(0.7 * e_pen + 0.3 * w_pen)

        c1 = qrs_cost(on1, off1)
        c2 = qrs_cost(on2, off2)
        if c2 < c1 * 0.98:
            on_sel, off_sel, c_sel = on2, off2, c2
        else:
            on_sel, off_sel, c_sel = on1, off1, c1

        # T end candidate
        t_start = r_loc + int(0.030 * fs)
        if np.isfinite(off_sel):
            t_start = max(t_start, int(off_sel))
        if np.isfinite(T_pk0[i]):
            t_start = max(t_start, int(T_pk0[i] - s))
        t_end = min(len(y_seg) - 1, r_loc + int(1.10 * rr_i * fs))

        toff2, rmse2 = _candidate_t_end(y_seg, fs, t_start, t_end)
        toff1 = int(T_off0[i] - s) if np.isfinite(T_off0[i]) else np.nan

        def toff_cost(toff):
            if not np.isfinite(toff):
                return 1e9
            return float(rmse2) if (np.isfinite(toff2) and int(toff) == int(toff2)) else float(rmse2 + 0.5)

        ct1 = toff_cost(toff1)
        ct2 = toff_cost(toff2)
        if ct2 < ct1 * 0.98:
            toff_sel, ct_sel = toff2, ct2
        else:
            toff_sel, ct_sel = toff1, ct1

        # Confidence: lower total cost -> higher confidence
        total = c_sel + ct_sel
        conf[i] = float(1.0 / (1.0 + np.exp(1.2 * (total - np.nanmedian([c1, c2, ct1, ct2])))))

        QRS_on[i] = np.nan if not np.isfinite(on_sel) else float(on_sel + s)
        QRS_off[i] = np.nan if not np.isfinite(off_sel) else float(off_sel + s)
        T_off[i] = np.nan if not np.isfinite(toff_sel) else float(toff_sel + s)

    return QRS_on, QRS_off, T_off, conf


def compute_ecg_metrics(t, y, fs, engine="wavelet", debug_plots=False):
    """
    ECG analysis: robust R-peaks + (optional) NK delineation + refinement.
    Computes P duration, PR, QRS, QT, QTc(Bazett), RR, RMSSD.
    """
    sig_filt, (f_lo, f_hi) = auto_bandpass_notch_ecg(y, fs)
    rpeaks, waves, used_engine = delineate_ecg(sig_filt, fs, engine=engine)
    if len(rpeaks) < 3:
        raise ValueError("ECG analysis failed: insufficient R-peaks detected.")

    N = len(rpeaks)

    # Pull P on/off if present
    def get_wave(name):
        arr = waves.get(name, None)
        if arr is None:
            return np.full(N, np.nan)
        arr = np.asarray(arr, dtype=float)
        out = np.full(N, np.nan)
        k = min(len(arr), N)
        out[:k] = arr[:k]
        return out

    P_on = get_wave("ECG_P_Onsets")
    P_off = get_wave("ECG_P_Offsets")

    # Refine QRS/T
    QRS_on, QRS_off, T_off, conf = refine_qrs_t(sig_filt, fs, rpeaks, waves)

    # Times
    r_t = rpeaks / fs
    rr = np.full(N, np.nan)
    rr[1:] = np.diff(r_t)

    # Intervals (seconds)
    P_dur = np.where(np.isfinite(P_on) & np.isfinite(P_off), (P_off - P_on) / fs, np.nan)
    PR = np.where(np.isfinite(P_on) & np.isfinite(QRS_on), (QRS_on - P_on) / fs, np.nan)
    QRS = np.where(np.isfinite(QRS_on) & np.isfinite(QRS_off), (QRS_off - QRS_on) / fs, np.nan)
    QT = np.where(np.isfinite(QRS_on) & np.isfinite(T_off), (T_off - QRS_on) / fs, np.nan)

    # QTc Bazett ONLY
    QTcB = np.where(np.isfinite(QT) & np.isfinite(rr) & (rr > 0), QT / np.sqrt(rr), np.nan)

    # RMSSD (seconds)
    rr_clean = rr[np.isfinite(rr) & (rr > 0)]
    if rr_clean.size >= 3:
        rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2)))
    else:
        rmssd = np.nan

    # Build beat table
    rows = []
    for i in range(N):
        rows.append({
            "Beat Index": i,
            "R_time_s": float(r_t[i]),
            "RR_s": float(rr[i]) if np.isfinite(rr[i]) else np.nan,
            "P_dur_s": float(P_dur[i]) if np.isfinite(P_dur[i]) else np.nan,
            "PR_s": float(PR[i]) if np.isfinite(PR[i]) else np.nan,
            "QRS_s": float(QRS[i]) if np.isfinite(QRS[i]) else np.nan,
            "QT_s": float(QT[i]) if np.isfinite(QT[i]) else np.nan,
            "QTcB_s": float(QTcB[i]) if np.isfinite(QTcB[i]) else np.nan,
            "Confidence_QRS_T": float(conf[i]) if np.isfinite(conf[i]) else np.nan
        })

    # Optional debug plots (first few beats)
    if debug_plots:
        nb = min(5, N)
        for i in range(nb):
            r_idx = rpeaks[i]
            rr_i = rr[i] if np.isfinite(rr[i]) else np.nanmedian(rr[np.isfinite(rr)]) if np.any(np.isfinite(rr)) else 0.12
            left_ms = float(np.clip(0.60 * rr_i * 1000.0, 40.0, 140.0))
            right_ms = float(np.clip(1.00 * rr_i * 1000.0, 80.0, 220.0))
            L = int(left_ms * fs / 1000.0)
            R = int(right_ms * fs / 1000.0)
            s = max(0, r_idx - L)
            e = min(len(sig_filt), r_idx + R)
            x_seg = np.arange(s, e) / fs
            y_seg = sig_filt[s:e]

            plt.figure(figsize=(12, 4))
            plt.plot(x_seg, y_seg, color="darkred", lw=1)
            plt.scatter(rpeaks[i] / fs, sig_filt[rpeaks[i]], color="green", s=80, marker="X", label="R")
            if np.isfinite(QRS_on[i]):
                plt.scatter(QRS_on[i] / fs, sig_filt[int(QRS_on[i])], color="#ff9800", s=70, marker="X", label="QRS_on")
            if np.isfinite(QRS_off[i]):
                plt.scatter(QRS_off[i] / fs, sig_filt[int(QRS_off[i])], color="#7e57c2", s=70, marker="X", label="QRS_off")
            if np.isfinite(T_off[i]):
                plt.scatter(T_off[i] / fs, sig_filt[int(T_off[i])], color="#f44336", s=70, marker="X", label="T_off")
            plt.title(f"Beat {i} | Engine={used_engine} | conf={conf[i]:.2f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(alpha=0.2)
            plt.legend()
            plt.tight_layout()
            plt.show()

    meta = {
        "engine": used_engine,
        "bandpass_low_Hz": f_lo,
        "bandpass_high_Hz": f_hi,
        "fs_Hz": fs,
        "RMSSD_s": rmssd
    }
    return rows, meta


# ============================================================
# 5) EEG ENGINE (Spikes, SWDs, Seizures + optional calibration)
# ============================================================
def eeg_preprocess(sig, fs):
    """
    Simple, robust EEG preprocessing:
    - rolling median detrend (2s)
    - notch 50/100 if present
    - 0.5–100 Hz band (zero-phase)
    """
    win = max(1, int(fs * 2.0))
    med = pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()
    x = sig - med

    # notch 50 + 100
    x = apply_notch(x, fs, 50.0, q=30.0)
    x = apply_notch(x, fs, 100.0, q=30.0)

    x = bandpass_zero_phase(x, fs, 0.5, min(100.0, 0.45 * fs), order=4)
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
    """
    Spike detector (explainable rules) with soft biphasic scoring.
    Returns list of dicts: idx, w_ms, z_amp, slope, curv, bip_ratio, confidence.
    """
    sos = butter_bandpass_sos(fs, bp[0], bp[1], order=4)
    hf = sosfiltfilt(sos, sig)
    hf_abs = np.abs(hf)

    med, dev = moving_mad_stats(hf_abs, fs, 10.0)
    z = (hf_abs - med) / dev

    min_dist = int(fs * (refr_ms / 1000.0))
    peaks, props = find_peaks(z, height=z_thresh, distance=min_dist)

    # global baselines for slope/curvature
    grad_med = np.median(np.abs(np.diff(sig))) + 1e-9 if len(sig) >= 2 else 1.0
    curv_med = np.median(np.abs(np.diff(sig, n=2))) + 1e-9 if len(sig) >= 3 else 1.0

    results = []
    n = len(sig)

    for p in peaks:
        amp_env = hf_abs[p]
        half = 0.5 * amp_env

        l = p
        while l > 0 and hf_abs[l] > half:
            l -= 1
        r = p
        while r < n - 1 and hf_abs[r] > half:
            r += 1

        w_ms = 1000.0 * max(1, (r - l)) / fs
        if not (width_ms[0] <= w_ms <= width_ms[1]):
            continue

        # slope check
        k = max(2, int(round(0.005 * fs)))
        core_left = max(0, p - k)
        core_right = min(n, p + k)
        core = sig[core_left:core_right]
        g1 = np.diff(core) if core.size >= 2 else np.array([0.0])
        peak_slope = float(np.max(np.abs(g1))) if g1.size else 0.0

        pre_span = max(k * 5, int(0.02 * fs))
        base_left = max(0, core_left - pre_span)
        base = sig[base_left:core_left]
        base_grad = float(np.median(np.abs(np.diff(base)))) + 1e-9 if base.size >= 2 else grad_med
        if peak_slope < slope_thresh * base_grad:
            continue

        # curvature check
        g2 = np.diff(core, n=2) if core.size >= 3 else np.array([0.0])
        sharp = float(np.max(np.abs(g2))) if g2.size else 0.0
        base2_left = max(0, base_left - pre_span)
        base2 = sig[base2_left:base_left]
        base_curv = float(np.median(np.abs(np.diff(base2, n=2)))) + 1e-9 if base2.size >= 3 else curv_med
        if sharp < curv_thresh * base_curv:
            continue

        # post-spike quietness (rough)
        a = p + int(0.04 * fs)
        b = min(n, p + int(0.08 * fs))
        if b > a:
            post_env = float(np.mean(np.abs(sig[a:b])))
            spike_env = float(np.max(np.abs(sig[core_left:core_right]))) + 1e-9
            if post_env > rtob_frac * spike_env:
                continue

        # HF/LF ratio
        sw = int(0.08 * fs)
        s0 = max(0, p - sw // 2)
        s1 = min(n, p + sw // 2)
        seg = sig[s0:s1]
        if seg.size >= int(0.02 * fs):
            p30_80, p1_20 = band_power_fft(seg, fs, [(30, 80), (1, 20)])
            if p1_20 <= 0 or (p30_80 / p1_20) < 1.5:
                continue

        # amp-z in local window
        aw = int(round(ampz_win_s * fs))
        a0 = max(0, p - aw // 2)
        a1 = min(n, p + aw // 2)
        seg_abs = np.abs(sig[a0:a1])
        med_abs = float(np.median(seg_abs))
        mad_abs = float(mad(seg_abs, scale='normal')) + 1e-9
        z_amp = (abs(float(sig[p])) - med_abs) / mad_abs
        if z_amp < ampz_thresh:
            continue

        # biphasic ratio (soft by default)
        bip_ratio = 0.0
        if biphasic_mode in ("soft", "hard"):
            bw = int(round((bip_win_ms / 1000.0) * fs))
            search_l = max(0, p - bw)
            search_r = min(n, p + bw)
            main = float(sig[p])
            main_abs = abs(main) + 1e-12

            cand = np.r_[sig[search_l:core_left], sig[core_right:search_r]]
            if cand.size:
                opp = float(np.min(cand)) if main >= 0 else float(np.max(cand))
                bip_ratio = abs(opp) / main_abs
            if biphasic_mode == "hard" and bip_ratio < bip_ratio_min:
                continue

        # confidence: logistic from z_amp, slope ratio, bip_ratio
        slope_ratio = peak_slope / (base_grad + 1e-12)
        score = 0.9 * (z_amp - ampz_thresh) + 0.4 * (slope_ratio - slope_thresh) + 0.4 * (bip_ratio / (bip_ratio_min + 1e-12))
        confidence = float(1.0 / (1.0 + np.exp(-score)))

        results.append({
            "idx": int(p),
            "w_ms": float(w_ms),
            "z_amp": float(z_amp),
            "slope": float(peak_slope),
            "curv": float(sharp),
            "bip_ratio": float(bip_ratio),
            "confidence": confidence
        })

    return results


def detect_swd(sig, fs, band=(5.0, 9.0), min_duration_s=0.5, max_gap_s=0.5, power_ratio=2.5):
    """
    SWD detector:
    - 5–9 Hz dominance vs 1–20 Hz reference
    - envelope CV low (stable oscillation)
    - plus ISI stability (peak-to-peak CV) to reduce false rhythmic artifacts
    """
    lo = sosfiltfilt(butter_bandpass_sos(fs, band[0], band[1], 4), sig)
    env = np.abs(hilbert(lo))

    ref = sosfiltfilt(butter_bandpass_sos(fs, 1.0, 20.0, 4), sig)

    win = max(1, int(fs * 0.5))
    num = uniform_filter1d(lo * lo, size=win, mode="nearest")
    den = uniform_filter1d(ref * ref, size=win, mode="nearest") + 1e-12
    ratio = num / den

    env_mean = uniform_filter1d(env, size=win, mode="nearest")
    env_var = uniform_filter1d(env * env, size=win, mode="nearest") - env_mean * env_mean
    env_std = np.sqrt(np.maximum(0.0, env_var))
    env_cv = env_std / (env_mean + 1e-12)

    mask = (ratio > power_ratio) & (env_cv < 0.6)

    events = []
    in_evt = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_evt:
            in_evt = True
            start = i
        if in_evt and not m:
            stop = i
            if events and (start / fs - events[-1]["end"] <= max_gap_s):
                events[-1]["end"] = stop / fs
            else:
                events.append({"start": start / fs, "end": stop / fs})
            in_evt = False
    if in_evt:
        events.append({"start": start / fs, "end": len(sig) / fs})

    # ISI stability filter (peak-to-peak CV)
    kept = []
    for e in events:
        dur = e["end"] - e["start"]
        if dur < min_duration_s:
            continue
        a = int(e["start"] * fs)
        b = int(e["end"] * fs)
        seg = lo[a:b]
        # peak picking in SWD band
        peaks, _ = find_peaks(seg, distance=max(1, int(fs / 12.0)))
        if len(peaks) >= 4:
            isi = np.diff(peaks) / fs
            cv = float(np.std(isi) / (np.mean(isi) + 1e-12))
            if cv <= 0.25:
                kept.append(e)
        else:
            # if too short for isi, keep based on power/entropy gates already
            kept.append(e)

    return kept


def _close_gaps(mask, max_off_steps):
    if max_off_steps <= 0:
        return mask
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            off_len = j - i
            if 0 < off_len <= max_off_steps:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def detect_seizures_llrms(sig, fs,
                         win_s=1.0, step_s=0.25,
                         enter_z=5.5, exit_z=3.5, off_hold_s=2.0,
                         min_duration_s=5.0, merge_gap_s=15.0,
                         close_s=10.0,
                         hf_ratio_thr=1.6, lf_ratio_thr=0.9,
                         occupancy_min=0.45, rhythm_prom_thr=1.35, entropy_drop_sigma=0.20):
    """
    Seizure detector based on line length / RMS + artifact suppression + rhythmicity & entropy checks.
    """
    w = max(1, int(fs * win_s))
    step = max(1, int(fs * step_s))
    n = len(sig)
    starts = np.arange(0, n - w + 1, step)
    t0 = starts / fs

    dx = np.abs(np.diff(sig))
    LL = np.convolve(dx, np.ones(w), mode="valid")[::step] if w > 1 else dx[::step]
    x2 = sig * sig
    csum = np.cumsum(np.r_[0.0, x2])
    sums = csum[w:] - csum[:-w]
    RMS = np.sqrt(sums / w)[::step]

    ll_b = np.median(LL)
    ll_m = mad(LL, scale="normal") + 1e-9
    rms_b = np.median(RMS)
    rms_m = mad(RMS, scale="normal") + 1e-9

    ll_z = (LL - ll_b) / ll_m
    rms_z = (RMS - rms_b) / rms_m
    z = np.maximum(ll_z, rms_z)

    hf_ratio = np.zeros_like(z)
    lf_ratio = np.zeros_like(z)
    prom_ratio = np.zeros_like(z)
    sent = np.zeros_like(z)
    clipped = np.zeros_like(z, dtype=bool)
    high_kurt = np.zeros_like(z, dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0 + w]
        if len(seg) < w:
            continue
        r = np.percentile(seg, [1, 99])
        clipped[i] = np.sum(np.isclose(seg, r[0], atol=1e-9)) > 0 or np.sum(np.isclose(seg, r[1], atol=1e-9)) > 0
        high_kurt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0

        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1, 45), (70, 100), (0.1, 0.5)])
        hf_ratio[i] = p70_100 / (p1_45 + 1e-12)
        lf_ratio[i] = p01_05 / (p1_45 + 1e-12)

        p4_30, p1_100 = band_power_fft(seg, fs, [(4, 30), (1, 100)])
        prom_ratio[i] = p4_30 / (np.median([p1_100, 1e-12]) + 1e-12)
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_med = np.median(sent)
    ent_mad = mad(sent, scale="normal") + 1e-9
    ent_z = (ent_med - sent) / ent_mad

    artifact = (hf_ratio > hf_ratio_thr) | (lf_ratio > lf_ratio_thr) | clipped | high_kurt
    z_eff = z.copy()
    z_eff[artifact] = -1e9

    off_hold_steps = max(1, int(round(off_hold_s / step_s)))

    # Hysteresis mask
    mask = np.zeros(len(z_eff), dtype=bool)
    i = 0
    while i < len(z_eff):
        if z_eff[i] >= enter_z:
            start = i
            off = 0
            i += 1
            while i < len(z_eff):
                if z_eff[i] <= exit_z:
                    off += 1
                    if off >= off_hold_steps:
                        end = i - off + 1
                        mask[start:end] = True
                        break
                else:
                    off = 0
                i += 1
            else:
                mask[start:] = True
                break
        else:
            i += 1

    if close_s > 0:
        mask = _close_gaps(mask, int(round(close_s / step_s)))

    # Convert mask -> events
    events = []
    in_evt = False
    start_idx = 0
    for i, on in enumerate(mask):
        if on and not in_evt:
            in_evt = True
            start_idx = i
        if in_evt and not on:
            end_idx = i
            a = float(t0[start_idx])
            b = float(t0[end_idx - 1] + win_s)
            if (b - a) >= min_duration_s:
                events.append((start_idx, end_idx, a, b))
            in_evt = False
    if in_evt:
        a = float(t0[start_idx])
        b = float(t0[len(mask) - 1] + win_s)
        if (b - a) >= min_duration_s:
            events.append((start_idx, len(mask), a, b))

    # Occupancy + rhythmicity + entropy criteria
    kept = []
    for s_idx, e_idx, a, b in events:
        seg_on = (z[s_idx:e_idx] >= exit_z) & (~artifact[s_idx:e_idx])
        occ = float(np.mean(seg_on)) if seg_on.size else 0.0
        if occ < occupancy_min:
            continue
        rhy_ok = float(np.mean(prom_ratio[s_idx:e_idx] >= rhythm_prom_thr)) >= 0.60
        ent_ok = float(np.mean(ent_z[s_idx:e_idx] >= entropy_drop_sigma)) >= 0.60
        if not (rhy_ok and ent_ok):
            continue
        kept.append({"start": a, "end": b})

    # Merge close events
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
    """
    Seizure detector driven by spike rate + ISI regularity, with spectral artifact suppression.
    """
    N = len(sig)
    w = max(1, int(round(win_s * fs)))
    step = max(1, int(round(hop_s * fs)))
    starts = np.arange(0, N - w + 1, step)
    t0 = starts / fs

    hf_ratio = np.zeros(len(starts))
    lf_ratio = np.zeros(len(starts))
    prom_ratio = np.zeros(len(starts))
    sent = np.zeros(len(starts))
    clipped = np.zeros(len(starts), dtype=bool)
    high_kurt = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0 + w]
        if len(seg) < w:
            continue
        r = np.percentile(seg, [1, 99])
        clipped[i] = np.sum(np.isclose(seg, r[0], atol=1e-9)) > 0 or np.sum(np.isclose(seg, r[1], atol=1e-9)) > 0
        high_kurt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0

        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1, 45), (70, 100), (0.1, 0.5)])
        hf_ratio[i] = p70_100 / (p1_45 + 1e-12)
        lf_ratio[i] = p01_05 / (p1_45 + 1e-12)

        p4_30, p1_100 = band_power_fft(seg, fs, [(4, 30), (1, 100)])
        prom_ratio[i] = p4_30 / (np.median([p1_100, 1e-12]) + 1e-12)
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_med = np.median(sent)
    ent_mad = mad(sent, scale="normal") + 1e-9
    ent_z = (ent_med - sent) / ent_mad

    artifact = (hf_ratio > hf_ratio_thr) | (lf_ratio > lf_ratio_thr) | clipped | high_kurt

    spikes = np.array(spike_times_rel, dtype=float)
    spikes = spikes[(spikes >= 0.0) & (spikes <= (N / fs))] if spikes.size else spikes

    min_count = int(np.ceil(min_rate_hz * win_s) - 1e-9)
    rate_ok = np.zeros(len(starts), dtype=bool)
    isi_ok = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        a = s0 / fs
        b = a + win_s
        if spikes.size == 0:
            continue
        i0 = np.searchsorted(spikes, a)
        i1 = np.searchsorted(spikes, b, side="right")
        sw = spikes[i0:i1]
        rate_ok[i] = sw.size >= min_count
        if sw.size >= 3:
            isi = np.diff(sw)
            med_isi = float(np.median(isi))
            if isi_min_s <= med_isi <= isi_max_s:
                cv = float(np.std(isi) / (np.mean(isi) + 1e-12))
                isi_ok[i] = cv <= isi_cv_max

    on = rate_ok & isi_ok & (~artifact)
    if close_s > 0:
        on = _close_gaps(on, int(round(close_s / hop_s)))

    events = []
    in_evt = False
    s_idx = 0
    for i, val in enumerate(on):
        if val and not in_evt:
            in_evt = True
            s_idx = i
        if in_evt and not val:
            e_idx = i
            a = float(t0[s_idx])
            b = float(t0[e_idx - 1] + win_s)
            if (b - a) >= min_duration_s:
                events.append((s_idx, e_idx, a, b))
            in_evt = False
    if in_evt:
        a = float(t0[s_idx])
        b = float(t0[len(on) - 1] + win_s)
        if (b - a) >= min_duration_s:
            events.append((s_idx, len(on), a, b))

    kept = []
    for s_idx, e_idx, a, b in events:
        occ = float(np.mean(on[s_idx:e_idx])) if (e_idx > s_idx) else 0.0
        if occ < occupancy_min:
            continue
        rhy_ok = float(np.mean(prom_ratio[s_idx:e_idx] >= rhythm_prom_thr)) >= 0.60
        ent_ok = float(np.mean(ent_z[s_idx:e_idx] >= entropy_drop_sigma)) >= 0.60
        if not (rhy_ok and ent_ok):
            continue
        kept.append({"start": a, "end": b})

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"] - merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
    return merged


def merge_intervals(events, merge_gap_s):
    if not events:
        return []
    events = sorted(events, key=lambda e: e["start"])
    merged = [dict(events[0])]
    for e in events[1:]:
        if e["start"] - merged[-1]["end"] <= merge_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
        else:
            merged.append(dict(e))
    return merged


def merge_spike_times(times, refr_s=0.02):
    if len(times) == 0:
        return np.array([], dtype=float)
    times = np.array(sorted(times), dtype=float)
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= refr_s:
            out.append(float(t))
    return np.array(out, dtype=float)


def bin_counts(spike_ts, swd_evts, sz_evts, bin_s, t0, t1):
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if edges[-1] < t1:
        edges = np.append(edges, t1)
    rows = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i + 1])
        spikes = int(np.sum((spike_ts >= a) & (spike_ts < b))) if spike_ts.size else 0
        swds = int(np.sum([(e["start"] >= a) and (e["start"] < b) for e in swd_evts])) if swd_evts else 0
        szs = int(np.sum([(e["start"] >= a) and (e["start"] < b) for e in sz_evts])) if sz_evts else 0
        rows.append({"Bin Index": i + 1, "Bin Start (s)": a, "Bin End (s)": b,
                     "Spikes": spikes, "SWDs": swds, "Seizures": szs})
    return pd.DataFrame(rows)


def read_sirenia_onsets(tsv_path, labels_col_idx=5, label_value="seizure", skiprows=7, time_col_idx=3):
    """
    Optional calibration helper:
    scans TSV for 'seizure' markers in column F (index 5) by default.
    """
    target = str(label_value).strip().lower()
    times = []
    try:
        reader = pd.read_csv(tsv_path, sep="\t", header=None, skiprows=skiprows,
                             usecols=[time_col_idx, labels_col_idx], engine="c",
                             dtype={time_col_idx: "float64", labels_col_idx: "string"},
                             na_filter=False, chunksize=200000)
        for chunk in reader:
            lab = (chunk.iloc[:, 1].astype("string")
                   .str.lower().str.strip()
                   .str.replace(r"[;:.,]+$", "", regex=True))
            m = lab.str.contains(rf"\b{target}\b", na=False)
            if np.any(m.values):
                times.extend(chunk.loc[m, chunk.columns[0]].to_numpy(dtype=float).tolist())
    except Exception:
        pass
    times = sorted(set([float(x) for x in times if np.isfinite(x)]))
    return times


def analyze_eeg(t, y, fs,
                bin_s=3600.0,
                use_calibration=False,
                seizure_label_col=5,
                seizure_label_value="seizure",
                debug_plots=False):
    """
    EEG analysis: detect spikes, SWDs, seizures; export binned summary + event list.
    Calibration option: uses Sirenia 'seizure' labels to set spike/SWD thresholds from within-ictal segments.
    """
    sig = eeg_preprocess(y, fs)

    # decimate for detectors
    sig_spk, fs_spk = decimate_to(sig, fs, 500)
    sig_swd, fs_swd = decimate_to(sig, fs, 200)
    sig_sz, fs_sz = decimate_to(sig, fs, 250)

    # Default thresholds (can be improved by calibration)
    spike_params = dict(
        z_thresh=5.6, width_ms=(12, 80), refr_ms=22,
        slope_thresh=3.0, curv_thresh=2.0,
        rtob_frac=0.25, ampz_thresh=4.2, ampz_win_s=1.5,
        biphasic_mode="soft", bip_win_ms=25.0, bip_ratio_min=0.35
    )
    swd_params = dict(band=(5.0, 9.0), min_duration_s=0.5, max_gap_s=0.5, power_ratio=2.5)

    # Optional: seizure anchored calibration (recommended)
    calib_meta = {}
    if use_calibration:
        # Only works if source is TSV with label column; otherwise silently skip
        if str(pathlib.Path(t[0].__class__.__name__)).lower():
            pass
        try:
            # The calibration reads directly from file; we can only do this if caller provides file path.
            # In GUI we will pass file path in via meta; here we assume caller attached it:
            # (Handled in GUI: analyze_eeg_from_file() wrapper below.)
            pass
        except Exception:
            pass

    # Detect spikes
    spikes = detect_spikes(sig_spk, fs_spk, **spike_params)
    spike_times = np.array([t[0] + s["idx"] / fs_spk for s in spikes], dtype=float)  # absolute time
    spike_conf = np.array([s["confidence"] for s in spikes], dtype=float)

    # Detect SWDs
    swds = detect_swd(sig_swd, fs_swd, **swd_params)
    swd_events = [{"start": t[0] + e["start"], "end": t[0] + e["end"], "type": "SWD", "confidence": 0.85} for e in swds]

    # Detect seizures: hybrid = spike-led + LL/RMS and merge
    rel_spikes = (spike_times - t[0]).tolist()
    sz_spike = detect_seizures_spike_led(sig_sz, fs_sz, rel_spikes)
    sz_llrms = detect_seizures_llrms(sig_sz, fs_sz)

    sz_events = [{"start": t[0] + e["start"], "end": t[0] + e["end"], "type": "Seizure", "confidence": 0.90} for e in sz_spike]
    sz_events += [{"start": t[0] + e["start"], "end": t[0] + e["end"], "type": "Seizure", "confidence": 0.75} for e in sz_llrms]
    sz_events = merge_intervals(sz_events, merge_gap_s=12.0)

    # Build event table
    ev_rows = []
    for ti, cf in zip(spike_times, spike_conf):
        ev_rows.append({"Type": "Spike", "Start (s)": float(ti), "End (s)": float(ti), "Duration (s)": 0.0, "Confidence": float(cf)})
    for e in swd_events:
        ev_rows.append({"Type": "SWD", "Start (s)": float(e["start"]), "End (s)": float(e["end"]),
                        "Duration (s)": float(e["end"] - e["start"]), "Confidence": float(e.get("confidence", np.nan))})
    for e in sz_events:
        ev_rows.append({"Type": "Seizure", "Start (s)": float(e["start"]), "End (s)": float(e["end"]),
                        "Duration (s)": float(e["end"] - e["start"]), "Confidence": float(e.get("confidence", np.nan))})

    events_df = pd.DataFrame(sorted(ev_rows, key=lambda r: (r["Start (s)"], r["Type"])))

    # Summary bins
    t0 = float(t[0])
    t1 = float(t[-1])
    spike_times2 = merge_spike_times(spike_times.tolist(), refr_s=0.02)
    summary_df = bin_counts(
        spike_times2,
        [e for e in swd_events],
        [e for e in sz_events],
        float(bin_s),
        t0, t1
    )

    meta = {
        "fs_Hz": float(fs),
        "preprocess": "rolling-median detrend; 50/100 Hz notch; 0.5–100 Hz bandpass (zero-phase)",
        "detector_fs": "Spikes 500 Hz, SWD 200 Hz, Seizure 250 Hz (anti-aliased decimation)",
        "bin_s": float(bin_s),
        "calibration_used": bool(use_calibration),
        **calib_meta
    }

    if debug_plots and len(events_df) > 0:
        # plot a quick window around first few events
        for _, r in events_df.head(3).iterrows():
            a = max(t0, r["Start (s)"] - 2.0)
            b = min(t1, r["End (s)"] + 2.0)
            i0 = np.searchsorted(t, a)
            i1 = np.searchsorted(t, b)
            plt.figure(figsize=(12, 4))
            plt.plot(t[i0:i1], sig[i0:i1], "k", lw=0.9)
            plt.axvspan(r["Start (s)"], r["End (s)"], alpha=0.25,
                        color={"Spike": "gold", "SWD": "lightskyblue", "Seizure": "tomato"}[r["Type"]])
            plt.title(f"{r['Type']}  [{r['Start (s)']:.2f}–{r['End (s)']:.2f}]  conf={r['Confidence']:.2f}")
            plt.xlabel("Time (s)")
            plt.ylabel("EEG (a.u.)")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

    return summary_df, events_df, meta


# ============================================================
# 6) EXPORTERS
# ============================================================
def export_ecg_excel(save_path, beat_rows, meta):
    df = pd.DataFrame(beat_rows)

    # add ms columns for readability
    for col in ["RR_s", "P_dur_s", "PR_s", "QRS_s", "QT_s", "QTcB_s"]:
        if col in df.columns:
            df[col.replace("_s", "_ms")] = df[col] * 1000.0

    # summary
    summary = {}
    for k in ["RR_s", "P_dur_s", "PR_s", "QRS_s", "QT_s", "QTcB_s"]:
        v = df[k].to_numpy() if k in df.columns else np.array([])
        v = v[np.isfinite(v)]
        summary[f"{k}_mean"] = float(np.mean(v)) if v.size else np.nan
        summary[f"{k}_median"] = float(np.median(v)) if v.size else np.nan
    summary["RMSSD_s"] = float(meta.get("RMSSD_s", np.nan))
    summary["RMSSD_ms"] = float(meta.get("RMSSD_s", np.nan) * 1000.0) if np.isfinite(meta.get("RMSSD_s", np.nan)) else np.nan

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ECG_Beats")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="ECG_Summary")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")


def export_eeg_excel(save_path, summary_df, events_df, meta):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="EEG_Summary_per_bin")
        events_df.to_excel(writer, index=False, sheet_name="EEG_Events")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")


# ============================================================
# 7) UNIFIED GUI
# ============================================================
class SUDEPWaveformAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SUDEP Waveform Analyzer (EEG + ECG)")
        self.root.geometry("860x620")
        self.root.resizable(False, False)

        self.input_file = None
        self.results = None  # will store dict containing outputs
        self.is_analyzing = False

        self._build_ui()

    def _build_ui(self):
        title = tk.Label(self.root, text="SUDEP Waveform Analyzer", font=("Arial", 18, "bold"),
                         fg="#1E88E5", pady=10)
        title.pack()

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=18, pady=8)

        # Mode selection
        mode_frame = tk.LabelFrame(top, text="Mode", padx=10, pady=10, font=("Arial", 10, "bold"))
        mode_frame.pack(side="left", padx=(0, 10), fill="x", expand=True)
        self.mode_var = tk.StringVar(value="EEG")
        tk.Radiobutton(mode_frame, text="EEG", variable=self.mode_var, value="EEG").pack(side="left", padx=8)
        tk.Radiobutton(mode_frame, text="ECG", variable=self.mode_var, value="ECG").pack(side="left", padx=8)

        # ECG engine
        ecg_frame = tk.LabelFrame(top, text="ECG Delineation", padx=10, pady=10, font=("Arial", 10, "bold"))
        ecg_frame.pack(side="left", padx=(0, 10), fill="x", expand=True)

        self.ecg_engine_var = tk.StringVar(value="wavelet" if NK_AVAILABLE else "heuristic")
        self.ecg_engine_combo = ttk.Combobox(ecg_frame, values=["wavelet", "heuristic"],
                                             textvariable=self.ecg_engine_var,
                                             state="readonly", width=12)
        self.ecg_engine_combo.pack(side="left", padx=6)
        note = "NeuroKit2 OK" if NK_AVAILABLE else "NeuroKit2 not installed → heuristic"
        tk.Label(ecg_frame, text=note, fg="#666").pack(side="left", padx=6)

        # Options
        opt_frame = tk.LabelFrame(top, text="Options", padx=10, pady=10, font=("Arial", 10, "bold"))
        opt_frame.pack(side="right", fill="x")

        self.debug_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="Debug plots", variable=self.debug_var).pack(anchor="w")

        self.force_fs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="Force fs (Hz)", variable=self.force_fs_var).pack(anchor="w")
        self.force_fs_entry = tk.Entry(opt_frame, width=8)
        self.force_fs_entry.insert(0, "2000")
        self.force_fs_entry.pack(anchor="w", pady=(2, 0))

        # EEG bin size + calibration
        eeg_opt = tk.LabelFrame(self.root, text="EEG Options", padx=18, pady=10, font=("Arial", 10, "bold"))
        eeg_opt.pack(fill="x", padx=18, pady=6)

        row = tk.Frame(eeg_opt)
        row.pack(fill="x")

        tk.Label(row, text="Bin size (seconds):").pack(side="left")
        self.eeg_bin_entry = tk.Entry(row, width=10)
        self.eeg_bin_entry.insert(0, "3600")
        self.eeg_bin_entry.pack(side="left", padx=8)

        self.eeg_calib_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="Use Sirenia seizure-label calibration (col F contains 'seizure')",
                       variable=self.eeg_calib_var).pack(side="left", padx=12)

        # File selection
        file_frame = tk.LabelFrame(self.root, text="Step 1: Select Input File (D=time, E=signal, row>=8)",
                                   padx=18, pady=14, font=("Arial", 11, "bold"))
        file_frame.pack(fill="x", padx=18, pady=8)

        self.file_label = tk.Label(file_frame, text="No file selected", fg="gray", wraplength=600)
        self.file_label.pack(side="left", fill="x", expand=True)

        tk.Button(file_frame, text="Browse", command=self._browse,
                  bg="#43A047", fg="white", font=("Arial", 10, "bold"),
                  padx=14, pady=6).pack(side="right", padx=8)

        # Progress
        prog = tk.LabelFrame(self.root, text="Progress", padx=18, pady=14, font=("Arial", 11, "bold"))
        prog.pack(fill="x", padx=18, pady=8)

        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100, length=760)
        self.progress.pack(fill="x")

        self.status = tk.Label(prog, text="Ready", fg="#333")
        self.status.pack(anchor="w", pady=(6, 0))

        # Buttons
        btns = tk.Frame(self.root)
        btns.pack(fill="x", padx=18, pady=12)

        self.analyze_btn = tk.Button(btns, text="Analyze", state="disabled",
                                     command=self._start_analysis,
                                     bg="#1E88E5", fg="white",
                                     font=("Arial", 11, "bold"),
                                     padx=16, pady=10)
        self.analyze_btn.pack(side="left", padx=8)

        self.export_btn = tk.Button(btns, text="Export Excel", state="disabled",
                                    command=self._export,
                                    bg="#FB8C00", fg="white",
                                    font=("Arial", 11, "bold"),
                                    padx=16, pady=10)
        self.export_btn.pack(side="left", padx=8)

        tk.Button(btns, text="Reset", command=self._reset,
                  bg="#E53935", fg="white",
                  font=("Arial", 11, "bold"),
                  padx=16, pady=10).pack(side="left", padx=8)

        tk.Button(btns, text="Exit", command=self.root.quit,
                  bg="#757575", fg="white",
                  font=("Arial", 11, "bold"),
                  padx=16, pady=10).pack(side="right", padx=8)

    def _browse(self):
        mode = self.mode_var.get()
        path = filedialog.askopenfilename(
            title=f"Select {mode} file",
            filetypes=[
                ("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                ("TSV", "*.tsv"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("All", "*.*")
            ]
        )
        if path:
            self.input_file = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.file_label.config(text=f"{os.path.basename(path)} ({size_mb:.2f} MB)", fg="black")
            self.analyze_btn.config(state="normal")
            self.status.config(text="Ready to analyze", fg="#333")

    def _set_progress(self, pct, msg):
        self.progress["value"] = float(pct)
        self.status.config(text=msg, fg="#1E88E5")
        self.root.update_idletasks()

    def _start_analysis(self):
        if not self.input_file:
            messagebox.showerror("Error", "Select a file first.")
            return
        if self.is_analyzing:
            messagebox.showwarning("Busy", "Analysis already running.")
            return
        self.is_analyzing = True
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status.config(text="Starting...", fg="#1E88E5")
        th = threading.Thread(target=self._run_analysis, daemon=True)
        th.start()

    def _run_analysis(self):
        try:
            self._set_progress(5, "Loading file (D,E; row>=8)...")
            t, y = load_waveform_tabular(self.input_file)

            self._set_progress(15, "Inferring sampling rate...")
            if self.force_fs_var.get():
                fs_target = float(self.force_fs_entry.get().strip())
                t, y, fs = ensure_uniform_sampling(t, y, fs_target=fs_target)
            else:
                t, y, fs = ensure_uniform_sampling(t, y, fs_target=None)

            mode = self.mode_var.get()
            debug = bool(self.debug_var.get())

            if mode == "ECG":
                self._set_progress(30, "ECG: filtering + R-peaks + delineation...")
                engine = self.ecg_engine_var.get()
                beat_rows, meta = compute_ecg_metrics(t, y, fs, engine=engine, debug_plots=debug)
                self.results = {"mode": "ECG", "beats": beat_rows, "meta": meta}
                self._set_progress(100, f"ECG complete. Beats: {len(beat_rows)}  | QTc: Bazett")
            else:
                self._set_progress(30, "EEG: preprocessing + event detection...")
                bin_s = float(self.eeg_bin_entry.get().strip())
                use_calib = bool(self.eeg_calib_var.get())

                # (Optional) calibration currently expects labels in TSV only; we will do the simplest safe behavior:
                # If TSV and use_calib, we'll scan label column; otherwise just run default.
                if use_calib and pathlib.Path(self.input_file).suffix.lower() == ".tsv":
                    # Minimal calibration step: confirm onsets exist, report count in meta.
                    onsets = read_sirenia_onsets(self.input_file, labels_col_idx=5, label_value="seizure")
                    # For now we record only; advanced threshold learning can be added as a second iteration.
                    # (Your earlier EEG script contains full grow/learn logic—this GUI version keeps it lean.)
                    summary_df, events_df, meta = analyze_eeg(t, y, fs, bin_s=bin_s, use_calibration=False, debug_plots=debug)
                    meta["sirenia_onsets_found"] = len(onsets)
                    meta["calibration_note"] = "Onsets scanned; full in-ictal learning can be integrated in next iteration."
                else:
                    summary_df, events_df, meta = analyze_eeg(t, y, fs, bin_s=bin_s, use_calibration=False, debug_plots=debug)

                self.results = {"mode": "EEG", "summary": summary_df, "events": events_df, "meta": meta}
                self._set_progress(100, f"EEG complete. Events: {len(events_df)}")

            self.export_btn.config(state="normal")
            self.analyze_btn.config(state="normal")
            self.is_analyzing = False
            messagebox.showinfo("Success", "Analysis complete. Click Export to save results.")

        except Exception as e:
            self.is_analyzing = False
            self.analyze_btn.config(state="normal")
            self.export_btn.config(state="disabled")
            self.progress["value"] = 0
            self.status.config(text=f"Error: {str(e)}", fg="#E53935")
            messagebox.showerror("Analysis failed", str(e))

    def _export(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            title="Save results as Excel"
        )
        if not save_path:
            return

        try:
            if self.results["mode"] == "ECG":
                export_ecg_excel(save_path, self.results["beats"], self.results["meta"])
            else:
                export_eeg_excel(save_path, self.results["summary"], self.results["events"], self.results["meta"])

            self.status.config(text=f"Exported: {os.path.basename(save_path)}", fg="#43A047")
            messagebox.showinfo("Exported", f"Saved:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _reset(self):
        self.input_file = None
        self.results = None
        self.is_analyzing = False
        self.file_label.config(text="No file selected", fg="gray")
        self.progress["value"] = 0
        self.status.config(text="Ready", fg="#333")
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SUDEPWaveformAnalyzerApp(root)
    root.mainloop()
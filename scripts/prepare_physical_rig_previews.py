"""Generate cached raw waveforms and FFT spectral profiles for dashboard display."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

def main():
    conditions = {
        "NORMAL": "normal_02.csv",
        "IMBALANCE": "imbalance_02.csv",
        "MISALIGNMENT": "misalignment_02.csv",
    }
    
    sample_segments = []
    spectra = []
    
    for cond, fname in conditions.items():
        df = pd.read_csv(RAW_DIR / fname)
        # Take 500 samples (5 seconds) from stable middle region
        seg = df.iloc[5000:5500].copy().reset_index(drop=True)
        seg["condition"] = cond
        seg["recording"] = fname.replace(".csv", "")
        seg["time_s"] = (seg["timestamp_ms"] - seg["timestamp_ms"].iloc[0]) / 1000.0
        seg["vibration_magnitude"] = np.sqrt(seg["accel_x"]**2 + seg["accel_y"]**2 + seg["accel_z"]**2)
        sample_segments.append(seg)
        
        # Compute Welch PSD on vibration magnitude across the full recording
        vm_full = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2).to_numpy()
        freqs, psd = welch(vm_full, fs=100, nperseg=256)
        for f, p in zip(freqs, psd):
            spectra.append({
                "condition": cond,
                "frequency_hz": float(f),
                "psd_magnitude": float(p),
            })
            
    df_samples = pd.concat(sample_segments, ignore_index=True)
    df_samples.to_csv(PROCESSED_DIR / "physical_rig_raw_samples.csv", index=False)
    
    df_spec = pd.DataFrame(spectra)
    df_spec.to_csv(PROCESSED_DIR / "physical_rig_fft_spectra.csv", index=False)
    
    print(f"Saved {len(df_samples)} raw sample points to physical_rig_raw_samples.csv")
    print(f"Saved {len(df_spec)} FFT spectral points to physical_rig_fft_spectra.csv")

if __name__ == "__main__":
    main()

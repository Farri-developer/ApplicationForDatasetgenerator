import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks

# ================= FILTER =================
def bandpass_filter(data, low, high, fs):
    nyq = fs / 2
    if fs <= 0 or low >= high:
        return data
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# ================= EEG =================
def compute_band_powers(signal, fs):
    bands = {
        "Delta": (1,4),
        "Theta": (4,8),
        "Alpha": (8,13),
        "Beta": (13,30),
        "Gamma": (30,45)
    }

    f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), int(fs*2)))

    powers = {}
    for band,(low,high) in bands.items():
        mask = (f>=low) & (f<high)
        powers[band] = np.trapz(pxx[mask], f[mask]) if np.any(mask) else 0

    return powers

# ================= PPG =================
def extract_hrv(signal, fs):
    peaks,_ = find_peaks(signal, distance=max(int(fs*0.4),1))

    if len(peaks)<2:
        return 0,0,0,0

    rr = np.diff(peaks)/fs*1000

    hr = 60/(np.mean(rr)/1000+1e-6)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) if len(rr)>1 else 0
    pnn50 = np.sum(np.abs(np.diff(rr))>50)/(len(rr)+1e-6)*100

    return hr, sdnn, rmssd, pnn50

# ================= MAIN PROCESS =================
def process_folder(main_path):

    all_data = []

    for folder in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder)

        if not os.path.isdir(folder_path):
            continue

        try:
            eeg_path = os.path.join(folder_path, "eeg.csv")
            ppg_path = os.path.join(folder_path, "ppg.csv")
            bp_path  = os.path.join(folder_path, "bp.csv")

            if not (os.path.exists(eeg_path) and os.path.exists(ppg_path) and os.path.exists(bp_path)):
                continue

            eeg_df = pd.read_csv(eeg_path)
            ppg_df = pd.read_csv(ppg_path)
            bp_df  = pd.read_csv(bp_path)

            tcol = 'timestamp' if 'timestamp' in eeg_df.columns else 'lsl_timestamp'

            eeg_fs = len(eeg_df)/(eeg_df[tcol].iloc[-1]-eeg_df[tcol].iloc[0]+1e-6)
            ppg_fs = len(ppg_df)/(ppg_df[tcol].iloc[-1]-ppg_df[tcol].iloc[0]+1e-6)

            start = max(eeg_df[tcol].iloc[0], ppg_df[tcol].iloc[0])
            end   = min(eeg_df[tcol].iloc[-1], ppg_df[tcol].iloc[-1])

            WIN_SEC = 30
            num_windows = int((end-start)/WIN_SEC)

            for w in range(num_windows):

                ws = start + w*WIN_SEC
                we = ws + WIN_SEC

                eeg_win = eeg_df[(eeg_df[tcol]>=ws)&(eeg_df[tcol]<we)]
                ppg_win = ppg_df[(ppg_df[tcol]>=ws)&(ppg_df[tcol]<we)]

                if len(eeg_win)<10 or len(ppg_win)<10:
                    continue

                features = {}

                # ===== EEG =====
                eeg_vals = bandpass_filter(
                    eeg_win[['EEG1','EEG2','EEG3','EEG4']].values,
                    0.5,45,eeg_fs
                )

                beta_list = []
                alpha_list = []

                for ch in range(4):
                    p = compute_band_powers(eeg_vals[:,ch], eeg_fs)

                    for band in p:
                        features[f"EEG{ch+1}_{band}"] = p[band]

                    ratio = p['Beta']/(p['Alpha']+1e-6)
                    features[f"EEG{ch+1}_BetaAlpha"] = ratio

                    beta_list.append(p['Beta'])
                    alpha_list.append(p['Alpha'])

                features["SI"] = np.mean(beta_list)/(np.mean(alpha_list)+1e-6)

                # ===== PPG =====
                hr, sdnn, rmssd, pnn50 = extract_hrv(ppg_win['PPG1'].values, ppg_fs)

                features.update({
                    "HR":hr,
                    "SDNN":sdnn,
                    "RMSSD":rmssd,
                    "pNN50":pnn50
                })

                # ===== BP =====
                last = bp_df.iloc[-1]

                features["DeltaSYS"] = float(last.get("DeltaSYS",0))
                features["DeltaDIA"] = float(last.get("DeltaDIA",0))
                features["DeltaPulse"] = float(last.get("DeltaPulse",0))

                # ===== STRESS SCORE =====
                EEG_stress = np.mean([
                    features["EEG1_BetaAlpha"],
                    features["EEG2_BetaAlpha"],
                    features["EEG3_BetaAlpha"],
                    features["EEG4_BetaAlpha"]
                ])

                PPG_stress = (hr/(rmssd+1e-6)) + (1/(sdnn+1e-6))

                stress_score = (0.7*EEG_stress)+(0.3*PPG_stress)
                features["Stress_Score"] = stress_score

                all_data.append(features)

        except Exception as e:
            print("ERROR:", e)

    df = pd.DataFrame(all_data)

    # ===== LABELING =====
    low = df["Stress_Score"].quantile(0.33)
    high = df["Stress_Score"].quantile(0.66)

    def label(x):
        if x < low:
            return "Low"
        elif x < high:
            return "Medium"
        else:
            return "High"

    df["Stress_Label"] = df["Stress_Score"].apply(label)

    save_path = os.path.join(main_path, "merged_dataset.csv")
    df.to_csv(save_path, index=False)

    return save_path


# ================= GUI =================
def select_folder():
    path = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, path)

def run_process():
    path = entry.get()

    if not os.path.isdir(path):
        messagebox.showerror("Error", "Invalid Folder")
        return

    try:
        save_path = process_folder(path)
        messagebox.showinfo("Success", f"Dataset created:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("EEG + PPG + BP Dataset Generator")
root.geometry("500x200")

entry = tk.Entry(root, width=50)
entry.pack(pady=20)

btn1 = tk.Button(root, text="Select Folder", command=select_folder)
btn1.pack()

btn2 = tk.Button(root, text="Generate Dataset", command=run_process)
btn2.pack(pady=20)

root.mainloop()
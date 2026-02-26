import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from PySide6 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, welch, find_peaks

# ================= CONFIG =================
DEFAULT_DIR = r"D:\DataSet"
MIN_GAMMA_FS = 100
WIN_SEC = 30

# ================= EEG ADAPTIVE BASELINE =================
def eeg_adaptive_stress(beta_alpha_list, global_baseline):

    ratio = np.mean(beta_alpha_list)

    # Adaptive deviation from baseline
    if global_baseline == 0:
        global_baseline = ratio

    deviation = ratio / (global_baseline + 1e-6)

    if deviation < 0.9:
        return 0
    elif deviation < 1.1:
        return 1
    else:
        return 2

# ================= PPG ADAPTIVE STRESS MODEL =================
def ppg_adaptive_stress(hr, sdnn, rmssd, pnn50):

    # ---------- Heart Rate Stress ----------
    hr_stress = 0

    if hr > 110:
        hr_stress = 1
    elif hr > 95:
        hr_stress = 0.7
    elif hr > 80:
        hr_stress = 0.4
    else:
        hr_stress = 0.1

    # ---------- HRV Stress ----------
    sdnn_stress = 1 - min(sdnn / 150, 1)

    rmssd_stress = 1 - min(rmssd / 120, 1)

    pnn_stress = 1 - min(pnn50 / 60, 1)

    # ---------- Weighted Fusion ----------
    final_score = (
        0.4 * hr_stress +
        0.3 * sdnn_stress +
        0.2 * rmssd_stress +
        0.1 * pnn_stress
    )

    return round(final_score * 2, 2)   # Scale to 0–2 range
# ================= BP ONLY READ =================
def bp_read_features(delta_sys, delta_dia, delta_pulse):
    return delta_sys, delta_dia, delta_pulse

# ================= SIGNAL PROCESSING =================
def bandpass_filter(data, low, high, fs):

    if fs <= 0:
        return data

    nyq = fs / 2
    high = min(high, nyq - 1)

    if low >= high:
        return data

    b, a = butter(4, [low / nyq, high / nyq], btype='band')

    return filtfilt(b, a, data, axis=0)


def compute_band_powers(signal, fs, include_gamma=True):

    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30)
    }

    if include_gamma and fs >= MIN_GAMMA_FS:
        bands["Gamma"] = (30, 50)

    if len(signal) < int(fs * 2):
        return {b: 0.0 for b in bands}

    f, pxx = welch(
        signal,
        fs=fs,
        nperseg=min(len(signal), int(fs * 2))
    )

    powers = {}

    for band, (low, high) in bands.items():
        mask = (f >= low) & (f < high)
        powers[band] = np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else 0.0

    return powers

# ================= HRV EXTRACTION =================
def extract_ppg_hrv(signal, fs):

    if fs <= 0 or len(signal) < fs * 2:
        return 0, 0, 0, 0

    peaks, _ = find_peaks(signal, distance=max(int(fs * 0.4), 1))

    if len(peaks) < 2:
        return 0, 0, 0, 0

    rr = np.diff(peaks) / fs * 1000

    if len(rr) == 0:
        return 0, 0, 0, 0

    hr = 60 / (np.mean(rr) / 1000 + 1e-6)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) if len(rr) > 1 else 0
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / (len(rr)+1e-6) * 100

    return hr, sdnn, rmssd, pnn50

# ================= FINAL FUSION =================
def final_fusion_score(eeg_score, ppg_score, tlx_score):

    final_score = (
        0.25 * eeg_score +
        0.25 * ppg_score +
        0.50 * tlx_score
    )

    if final_score < 0.5:
        return final_score, 0
    elif final_score < 1:
        return final_score, 1
    else:
        return final_score, 2

# ================= GUI CLASS =================
class DatasetGenerator(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multimodal Stress Dataset Generator")
        self.setFixedSize(600, 500)

        self.eeg_file = ""
        self.ppg_file = ""
        self.bp_file = ""
        self.tlx_file = ""

        self.init_ui()

    # ---------- UI ----------
    def init_ui(self):

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Multimodal Stress Dataset Generator")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        group = QtWidgets.QGroupBox("Select Input Files")
        grid = QtWidgets.QGridLayout(group)

        self.lbl_eeg = QtWidgets.QLabel("EEG not selected")
        self.lbl_ppg = QtWidgets.QLabel("PPG not selected")
        self.lbl_bp = QtWidgets.QLabel("BP not selected")
        self.lbl_tlx = QtWidgets.QLabel("TLX not selected")

        buttons = [
            ("Select EEG", self.select_eeg, self.lbl_eeg),
            ("Select PPG", self.select_ppg, self.lbl_ppg),
            ("Select BP", self.select_bp, self.lbl_bp),
            ("Select TLX", self.select_tlx, self.lbl_tlx),
        ]

        for i, (text, func, label) in enumerate(buttons):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(func)

            grid.addWidget(btn, i, 0)
            grid.addWidget(label, i, 1)

        layout.addWidget(group)

        self.btn_generate = QtWidgets.QPushButton("Generate Final Dataset")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.generate_dataset)

        layout.addWidget(self.btn_generate)

    # ---------- File Selection ----------
    def update_btn(self):
        if all([self.eeg_file, self.ppg_file, self.bp_file, self.tlx_file]):
            self.btn_generate.setEnabled(True)

    def select_file(self, attr, label):

        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            DEFAULT_DIR,
            "CSV (*.csv)"
        )

        if f:
            setattr(self, attr, f)
            label.setText(os.path.basename(f))
            self.update_btn()

    def select_eeg(self): self.select_file("eeg_file", self.lbl_eeg)
    def select_ppg(self): self.select_file("ppg_file", self.lbl_ppg)
    def select_bp(self): self.select_file("bp_file", self.lbl_bp)
    def select_tlx(self): self.select_file("tlx_file", self.lbl_tlx)

    # ---------- Dataset Generation ----------
    def generate_dataset(self):

        try:
            eeg_df = pd.read_csv(self.eeg_file)
            ppg_df = pd.read_csv(self.ppg_file)
            bp_df = pd.read_csv(self.bp_file)
            tlx_df = pd.read_csv(self.tlx_file)

            # TLX Stress Label
            tlx_df['Dimension'] = tlx_df['Dimension'].astype(str).str.strip()

            stress_rows = tlx_df[
                tlx_df['Dimension'].str.lower() == 'stress_label'
            ]

            tlx_score = int(stress_rows['Score'].values[0]) if len(stress_rows) > 0 else 0

            # BP Read Only
            bp_df['label'] = bp_df['label'].astype(str).str.strip()

            question_rows = bp_df[bp_df['label'] == 'Question-End']

            last_row = question_rows.iloc[-1] if len(question_rows) > 0 else bp_df.iloc[-1]

            delta_sys = float(last_row.get('DeltaSYS', 0))
            delta_dia = float(last_row.get('DeltaDIA', 0))
            delta_pulse = float(last_row.get('DeltaPulse', 0))

            # Sampling Frequency
            eeg_time_range = eeg_df['lsl_timestamp'].iloc[-1] - eeg_df['lsl_timestamp'].iloc[0]
            ppg_time_range = ppg_df['lsl_timestamp'].iloc[-1] - ppg_df['lsl_timestamp'].iloc[0]

            eeg_fs = len(eeg_df) / (eeg_time_range + 1e-6)
            ppg_fs = len(ppg_df) / (ppg_time_range + 1e-6)

            include_gamma = eeg_fs >= MIN_GAMMA_FS

            rows = []

            start_time = max(
                eeg_df['lsl_timestamp'].iloc[0],
                ppg_df['lsl_timestamp'].iloc[0]
            )

            end_time = min(
                eeg_df['lsl_timestamp'].iloc[-1],
                ppg_df['lsl_timestamp'].iloc[-1]
            )

            num_windows = int((end_time - start_time) / WIN_SEC)

            # ===== Global EEG Baseline =====
            global_beta_alpha = []

            # Precompute global EEG baseline
            eeg_vals_global = bandpass_filter(
                eeg_df[['EEG1','EEG2','EEG3','EEG4']].values,
                0.5, 45, eeg_fs
            )

            for ch in range(4):

                powers = compute_band_powers(
                    eeg_vals_global[:, ch],
                    eeg_fs,
                    include_gamma
                )

                ratio = powers['Beta'] / (powers['Alpha'] + 1e-6)
                global_beta_alpha.append(ratio)

            global_baseline = np.mean(global_beta_alpha)

            # ===== Window Processing =====
            for w in range(num_windows):

                win_start = start_time + w * WIN_SEC
                win_end = win_start + WIN_SEC

                eeg_win = eeg_df[
                    (eeg_df['lsl_timestamp'] >= win_start) &
                    (eeg_df['lsl_timestamp'] < win_end)
                ]

                ppg_win = ppg_df[
                    (ppg_df['lsl_timestamp'] >= win_start) &
                    (ppg_df['lsl_timestamp'] < win_end)
                ]

                if len(eeg_win) < 10 or len(ppg_win) < 10:
                    continue

                features = {
                    "Time_Start": win_start,
                    "Time_End": win_end,
                    "Window_Index": w
                }

                # ===== EEG Feature =====
                eeg_vals = bandpass_filter(
                    eeg_win[['EEG1','EEG2','EEG3','EEG4']].values,
                    0.5, 45, eeg_fs
                )

                beta_alpha_list = []

                for ch in range(4):

                    powers = compute_band_powers(
                        eeg_vals[:, ch],
                        eeg_fs,
                        include_gamma
                    )

                    for band in powers:
                        features[f"EEG{ch+1}_{band}"] = powers[band]

                    ratio = powers['Beta'] / (powers['Alpha'] + 1e-6)

                    features[f"EEG{ch+1}_BetaAlpha"] = ratio

                    beta_alpha_list.append(ratio)

                eeg_score = eeg_adaptive_stress(
                    beta_alpha_list,
                    global_baseline
                )

                # ===== PPG Feature =====
                hr, sdnn, rmssd, pnn50 = extract_ppg_hrv(
                    ppg_win['PPG1'].values,
                    ppg_fs
                )

                ppg_score = ppg_adaptive_stress(
                    hr, sdnn, rmssd, pnn50
                )

                # ===== Fusion =====
                final_score, final_label = final_fusion_score(
                    eeg_score,
                    ppg_score,
                    tlx_score
                )

                features.update({
                    "HR": hr,
                    "SDNN": sdnn,
                    "RMSSD": rmssd,
                    "pNN50": pnn50,

                    "DeltaSYS": delta_sys,
                    "DeltaDIA": delta_dia,
                    "DeltaPulse": delta_pulse,

                    "EEG_Score": eeg_score,
                    "PPG_Score": ppg_score,

                    "TLX_Score": tlx_score,

                    "Fusion_Score": final_score,
                    "Final_Stress_Label": final_label
                })

                rows.append(features)

            df = pd.DataFrame(rows)

            save_dir = os.path.dirname(self.eeg_file)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_path = os.path.join(
                save_dir,
                f"final_dataset_{timestamp}.csv"
            )

            df.to_csv(save_path, index=False)

            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Dataset saved at:\n{save_path}\nTotal Windows: {len(df)}"
            )

        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

# ================= MAIN =================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = DatasetGenerator()
    window.show()

    sys.exit(app.exec())
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, folder_path)

def scan_and_generate_report():
    main_path = path_entry.get()

    if not os.path.isdir(main_path):
        messagebox.showerror("Error", "Select valid folder")
        return

    report_data = []

    for folder in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

        feature_file = None
        bp_file = None
        eeg_file = None
        ppg_file = None

        for f in files:
            name = f.lower()

            if "feature" in name and f.endswith(".csv"):
                feature_file = f
            elif "bp" == name.replace(".csv", ""):
                bp_file = f
            elif "eeg" == name.replace(".csv", ""):
                eeg_file = f
            elif "ppg" == name.replace(".csv", ""):
                ppg_file = f

        format_status = "OK"
        try:
            columns_set = None

            for f in [feature_file, bp_file, eeg_file, ppg_file]:
                if f:
                    df = pd.read_csv(os.path.join(folder_path, f))
                    cols = tuple(df.columns)

                    if columns_set is None:
                        columns_set = cols
                    else:
                        if cols != columns_set:
                            format_status = "Mismatch"
                            break

        except:
            format_status = "Error"

        report_data.append({
            "Folder": folder,
            "Feature File": feature_file if feature_file else "Missing",
            "BP": "Yes" if bp_file else "No",
            "EEG": "Yes" if eeg_file else "No",
            "PPG": "Yes" if ppg_file else "No",
            "Format Status": format_status
        })

    df_report = pd.DataFrame(report_data)
    save_path = os.path.join(main_path, "final_report.csv")
    df_report.to_csv(save_path, index=False)

    messagebox.showinfo("Done", f"Report Generated:\n{save_path}")


# 🔥 NEW FUNCTION
def merge_feature_files():
    main_path = path_entry.get()

    if not os.path.isdir(main_path):
        messagebox.showerror("Error", "Select valid folder")
        return

    all_data = []

    for folder in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder)

        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if "feature" in f.lower() and f.endswith(".csv"):
                file_path = os.path.join(folder_path, f)

                try:
                    df = pd.read_csv(file_path)

                    # add folder column
                    df["Folder"] = folder

                    all_data.append(df)

                except:
                    print(f"Error reading {file_path}")

    if not all_data:
        messagebox.showwarning("Warning", "No feature files found")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    # ✅ REQUIRED COLUMN ORDER
    required_columns = [

        "EEG1_Delta", "EEG1_Theta", "EEG1_Alpha", "EEG1_Beta", "EEG1_Gamma", "EEG1_BetaAlpha",
        "EEG2_Delta", "EEG2_Theta", "EEG2_Alpha", "EEG2_Beta", "EEG2_Gamma", "EEG2_BetaAlpha",
        "EEG3_Delta", "EEG3_Theta", "EEG3_Alpha", "EEG3_Beta", "EEG3_Gamma", "EEG3_BetaAlpha",
        "EEG4_Delta", "EEG4_Theta", "EEG4_Alpha", "EEG4_Beta", "EEG4_Gamma", "EEG4_BetaAlpha",
        "SI", "HR", "SDNN", "RMSSD", "pNN50",
        "DeltaSYS", "DeltaDIA", "DeltaPulse",
         "Stress_Label",
    ]

    # ✅ Keep only available columns (avoid crash if mismatch)
    available_columns = [col for col in required_columns if col in merged_df.columns]
    merged_df = merged_df[available_columns]

    # ✅ REMOVE ROWS WITH MISSING DATA
    merged_df = merged_df.dropna()

    # save file
    save_path = os.path.join(main_path, "merged_features_clean.csv")
    merged_df.to_csv(save_path, index=False)

    messagebox.showinfo("Done", f"Clean merged file saved:\n{save_path}")


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Advanced Dataset Scanner")
root.geometry("600x400")

tk.Label(root, text="Select Main Folder", font=("Arial", 12)).pack(pady=5)

frame = tk.Frame(root)
frame.pack(pady=5)

path_entry = tk.Entry(frame, width=50)
path_entry.pack(side=tk.LEFT, padx=5)

tk.Button(frame, text="Browse", command=select_folder).pack(side=tk.LEFT)

tk.Button(root, text="Scan + Generate Report",
          command=scan_and_generate_report,
          bg="blue", fg="white").pack(pady=10)

# 🔥 NEW BUTTON
tk.Button(root, text="Merge Feature Files",
          command=merge_feature_files,
          bg="green", fg="white").pack(pady=10)

root.mainloop()
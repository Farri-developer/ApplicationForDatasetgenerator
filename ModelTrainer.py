import sys
import os
import pandas as pd
import traceback
from PySide6 import QtWidgets, QtCore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class ModelTrainer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stress Model Trainer")
        self.setFixedSize(800,600)

        self.dataset_folder = ""

        self.init_ui()

    def init_ui(self):

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Random Forest Stress Model Trainer")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:20px;font-weight:bold")
        layout.addWidget(title)

        self.btn_select = QtWidgets.QPushButton("Select Main Dataset Folder")
        self.btn_select.clicked.connect(self.select_folder)
        layout.addWidget(self.btn_select)

        self.btn_train = QtWidgets.QPushButton("Train Model")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.train_model)
        layout.addWidget(self.btn_train)

        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def log(self,msg):
        self.output.append(msg)

    def select_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Main Folder"
        )

        if folder:
            self.dataset_folder = folder
            self.log(f"Selected Folder: {folder}")
            self.btn_train.setEnabled(True)

    def train_model(self):

        try:

            all_files = []

            # scan participant folders
            for root, dirs, files in os.walk(self.dataset_folder):

                for file in files:

                    if file.startswith("final_dataset") and file.endswith(".csv"):

                        path = os.path.join(root,file)
                        all_files.append(path)

            if len(all_files) == 0:
                self.log("No dataset files found")
                return

            self.log(f"Total dataset files found: {len(all_files)}")

            dfs = []

            for f in all_files:
                self.log(f"Loading: {f}")
                dfs.append(pd.read_csv(f))

            df = pd.concat(dfs,ignore_index=True)

            self.log(f"Total rows after combining: {len(df)}")

            # Remove leakage columns
            columns_to_remove = [
                "Time_Start",
                "Time_End",
                "Window_Index",
                "EEG_Score",
                "PPG_Score",
                "TLX_Score",
                "Fusion_Score"
            ]

            df = df.drop(columns=columns_to_remove,errors='ignore')

            self.log(f"Columns after cleaning:\n{list(df.columns)}")

            # features
            X = df.drop("Final_Stress_Label",axis=1)
            y = df["Final_Stress_Label"]

            X_train,X_test,y_train,y_test = train_test_split(
                X,y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            self.log(f"Training size: {len(X_train)}")
            self.log(f"Testing size: {len(X_test)}")

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight="balanced",
                random_state=42
            )

            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)

            self.log(f"\nTest Accuracy: {accuracy*100:.2f}%")

            report = classification_report(y_test,y_pred)
            self.log("\nClassification Report:\n"+report)

            cm = confusion_matrix(y_test,y_pred)
            self.log("\nConfusion Matrix:\n"+str(cm))

            cv_scores = cross_val_score(model,X,y,cv=5)

            self.log("\nCross Validation Scores: "+str(cv_scores))
            self.log(f"Average CV Accuracy: {cv_scores.mean()*100:.2f}%")

            importance = pd.DataFrame({
                "Feature":X.columns,
                "Importance":model.feature_importances_
            }).sort_values(by="Importance",ascending=False)

            self.log("\nTop Important Features:\n"+str(importance.head(10)))

            model_path = os.path.join(self.dataset_folder,"stress_model_random_forest.pkl")

            joblib.dump(model,model_path)

            self.log(f"\nModel saved at: {model_path}")

        except Exception as e:
            traceback.print_exc()
            self.log(str(e))


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    window = ModelTrainer()
    window.show()

    sys.exit(app.exec())
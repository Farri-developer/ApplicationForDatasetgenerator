import sys
import joblib
import pandas as pd
import numpy as np
from PySide6 import QtWidgets, QtCore


class StressPredictor(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stress Prediction System")
        self.setFixedSize(800,600)

        self.model = None
        self.data_file = ""

        self.init_ui()

    def init_ui(self):

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Stress Prediction Interface")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:20px;font-weight:bold")
        layout.addWidget(title)

        self.btn_model = QtWidgets.QPushButton("Load Trained Model")
        self.btn_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_model)

        self.btn_file = QtWidgets.QPushButton("Select Dataset File")
        self.btn_file.clicked.connect(self.select_file)
        layout.addWidget(self.btn_file)

        self.btn_predict = QtWidgets.QPushButton("Predict Stress")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.predict)
        layout.addWidget(self.btn_predict)

        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def log(self,msg):
        self.output.append(str(msg))

    def load_model(self):

        file,_ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "Model (*.pkl)"
        )

        if file:
            self.model = joblib.load(file)
            self.log(f"Model Loaded: {file}")

    def select_file(self):

        file,_ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV (*.csv)"
        )

        if file:
            self.data_file = file
            self.log(f"Dataset Selected: {file}")

        if self.model and self.data_file:
            self.btn_predict.setEnabled(True)

    def predict(self):

        try:

            df = pd.read_csv(self.data_file)

            self.log("\nOriginal Columns:")
            self.log(df.columns)

            columns_to_remove = [
                "Time_Start",
                "Time_End",
                "Window_Index",
                "EEG_Score",
                "PPG_Score",
                "TLX_Score",
                "Fusion_Score",
                "Final_Stress_Label"
            ]

            df_clean = df.drop(columns=columns_to_remove, errors="ignore")

            self.log("\nColumns used for prediction:")
            self.log(df_clean.columns)

            predictions = self.model.predict(df_clean)

            df["Predicted_Stress"] = predictions

            self.log("\nWindow Level Predictions:")
            self.log(predictions)

            vote_counts = pd.Series(predictions).value_counts()

            final_stress_majority = vote_counts.idxmax()

            self.log("\nVoting Result:")
            self.log(vote_counts)

            self.log(f"\nFinal Session Stress (Majority Voting): {final_stress_majority}")

            if hasattr(self.model,"predict_proba"):

                probs = self.model.predict_proba(df_clean)

                avg_probs = np.mean(probs,axis=0)

                final_stress_probability = np.argmax(avg_probs)

                confidence = max(avg_probs)*100

                self.log("\nAverage Class Probabilities:")
                self.log(avg_probs)

                self.log(f"\nFinal Session Stress (Probability Based): {final_stress_probability}")

                self.log(f"\nPrediction Confidence: {confidence:.2f}%")

            output_path = self.data_file.replace(".csv","_prediction.csv")

            df.to_csv(output_path,index=False)

            self.log(f"\nPrediction file saved at: {output_path}")

        except Exception as e:

            self.log(str(e))


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    window = StressPredictor()
    window.show()

    sys.exit(app.exec())
# SWV-Concentration-Prediction-ML
Machine learning–based prediction of analyte concentration from SWV curves responses of MIP-based electrochemical sensors.

---

## 🎯 Purpose of This Project

In electrochemical sensing (specifically MIP sensors), the baseline can be complex and non-linear. Instead of manually identifying peaks, this project utilizes three advanced architectures to map the raw $I/V$ relationship directly to concentration (nM).

***Model A: 1D-CNN (The Pattern Finder)** – Uses convolutional filters to detect local morphological features of the SWV peak.

***Model B: LSTM (The Sequence Learner)** – Processes the curve as a time-series to capture sequential dependencies in the potential scan.

***Model C: Hybrid CNN-Transformer (The Global Observer)** – Combines local feature extraction with self-attention mechanisms to observe global relationships across the entire scan.

--- 

## ▶️ How to Run

**Prepare Data**: Ensure your SWV curves are in the data/ directory.

**Training**: Run the specific model scripts: 1_1D_CNN.py; 2_LSTM_GRU.py; 3_Hybrid_CNN_Transformer.py

---

## 📊 Input Data

SWV curves for 5, 20, 50, 100, 150, 200, 250, 300, 350, 400 nM will be augmented to 200 curve for each concentration => total 2000 curve x 90 data points per curve x 2 variables measured at each step (Voltage and Current).
![calibration curve](https://github.com/user-attachments/assets/55f8e42c-0ba7-4a30-bd80-4c132056c93b)

---

## 📊 Result and Conclusion

All models achieved high accuracy ($R^2 > 0.99$), with the Hybrid architecture showing the superior precision required for low-concentration detection.

| Model | A | B | C |
| :--- | :---: | :---: | :---: |
| **Best epoch** | 49 | 51 | 307 |
| **R2** | 0.9966 | 0.9937 | 0.9987 |
| **MAE (nM)** | 5 | 7.74 | 2.97 |
| **RMSE (nM)** | 7.77 | 10.61 | 4.72 |

The results for the Hybrid CNN-Transformer model (Figures below) indicate a highly successful training phase and superior predictive accuracy for electrochemical concentration sensing.

**Loss Minimization**: The Model Loss (MSE) plot shows a rapid exponential decay in the early epochs, indicating that the hybrid architecture quickly identifies the fundamental relationship between the SWV curves and concentration.

**No Overfitting**: There is a perfect alignment between the Train Loss and Val Loss (as well as the MAE curves). The validation metrics do not diverge from the training metrics, proving that the model generalizes exceptionally well to unseen data and has not merely memorized the training set.

**Learning Efficiency**: The fluctuations in MAE settle significantly after epoch 100, reaching a stable plateau around epoch 307. This justifies the use of a higher epoch count for the Transformer's attention mechanism to fine-tune its weights.

**Linearity**: The Predicted vs. Actual plot shows that almost all data points lie directly on or very near the ideal $45^\circ$ dashed line. This confirms that the model maintains linear precision across the entire range from 5 nM to 400 nM.

**Consistency**: The tight clustering of points at each concentration level (e.g., at 100 nM or 300 nM) suggests low variance in the model's predictions. Even at the highest concentration (400 nM), where sensor saturation often occurs in traditional methods, Model C maintains high fidelity.

**Low Detection Limit Potential**: The high accuracy at the 5 nM and 20 nM points suggests the model is capable of distinguishing signal from background noise even at low analyte concentrations, which is a significant advantage of the Hybrid architecture.

<img width="1200" height="500" alt="Figure_C1" src="https://github.com/user-attachments/assets/794661d9-0ce5-49fb-aae4-e083cfccb45d" />

<img width="550" height="500" alt="Figure_C2" src="https://github.com/user-attachments/assets/9464afe4-b1ab-44ac-8660-bd43b631fe77" />

--- 

## 🧑‍💻 Author

Developed by: Vu Bao Chau Nguyen, Ph.D.

Keywords: MIP Sensors, Deep Learning, 1D-CNN, LSTM, Transformer, Electrochemical Sensing.

---

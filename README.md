# SWV-Concentration-Prediction-ML
Machine learning–based prediction of analyte concentration from simulated SWV curves of MIP-based electrochemical sensors.

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

**The data are simulated, not measured.** Simulated SWV curves for 5, 20, 50, 100, 150, 200, 250, 300, 350 and 400 nM were augmented to 200 curves per concentration => 2000 curves x 90 data points per curve x 2 variables at each step (voltage and current).
![calibration curve](https://github.com/user-attachments/assets/55f8e42c-0ba7-4a30-bd80-4c132056c93b)

---

## 📊 Result and Conclusion

On the held-out test set all three models reach $R^2 > 0.99$; the hybrid CNN–Transformer (Model C) gives the lowest error.

| Model | A | B | C |
| :--- | :---: | :---: | :---: |
| **Best epoch** | 49 | 51 | 307 |
| **R2** | 0.9966 | 0.9937 | 0.9987 |
| **MAE (nM)** | 5 | 7.74 | 2.97 |
| **RMSE (nM)** | 7.77 | 10.61 | 4.72 |

The figures below show training curves and predicted vs. actual concentration for Model C. Train and validation loss stay close throughout training, and the MAE settles after about epoch 100.

### What these results show, and what they don't

* **Same originals in training and test.** All 200 curves at a concentration are augmented copies of the same simulated original, and the 70/15/15 split is random. The test set therefore contains copies of curves the models were trained on, so the scores measure how well the models fit this synthetic dataset, not how they would perform on new measurements.
* **Concentration steps smaller than the augmentation spread.** At the high end the mean peak-to-baseline current (max − min of each curve) rises only about 5 % from 350 to 400 nM, while within each concentration it varies by about 11 % (coefficient of variation). Separating these levels with an RMSE of about 5 nM suggests the models also use features specific to each original curve, not only the signal size.
* **Train and validation loss agree** because both sets come from the same originals; this does not test generalization to new sensors or measurements.
* **No detection limit** can be estimated: the dataset has no blank or near-blank curves.

### Next steps for a fair test

* Hold out whole concentrations (for example, train without 150 nM and test on it).
* Simulate new original curves (new noise, baseline and drift) and use them only for testing.
* Validate on measured SWV curves from a MIP sensor.

<img width="1200" height="500" alt="Figure_C1" src="https://github.com/user-attachments/assets/794661d9-0ce5-49fb-aae4-e083cfccb45d" />

<img width="550" height="500" alt="Figure_C2" src="https://github.com/user-attachments/assets/9464afe4-b1ab-44ac-8660-bd43b631fe77" />

--- 

## 🧑‍💻 Author

Developed by: Vu Bao Chau Nguyen, Ph.D.

Keywords: MIP Sensors, Deep Learning, 1D-CNN, LSTM, Transformer, Electrochemical Sensing.

---

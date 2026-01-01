# SWV-Concentration-Prediction-ML
Machine learning–based prediction of analyte concentration from SWV curves responses of MIP-based electrochemical sensors.

--- updating ---
Ideal calibratuion curve, instead of using current peak values, we going to build ML model that using the whole SWV curveas input.
![calibration curve](https://github.com/user-attachments/assets/55f8e42c-0ba7-4a30-bd80-4c132056c93b)

# 3 models will be test:

**Model A: 1D-CNN (The Pattern Finder)**
This model uses "filters" that slide across the curve. It detects local features (like the start of a peak or a specific slope) regardless of where they appear on the voltage axis.

**Model B: LSTM/GRU (The Sequence Learner)**
This treats the SWV curve like a time-series. It processes the current values one by one, "remembering" what happened at the previous voltage step.

**Model C: Hybrid CNN-Transformer (The Global Observer)**
The CNN part extracts local features, and the Self-Attention mechanism of the Transformer looks at the entire curve at once to see how different parts of the scan relate to each other.

# Comparation metrics:
Mean Absolute Error, Mean Absolute Percentage Error, Correlation Coefficient

# Input data
SWV curves for 5, 20, 50, 100, 150, 200, 250, 300, 350, 400 nM will be augmented to 200 curve for each concentration => total 2000 curve x 90 data points per curve x 2 variables measured at each step (Voltage and Current).

# Results
| Model | A | B | C |
| :--- | :---: | :---: | :---: |
| **Best epoch** | 49 | 51 | 307 |
| **R2** | 0.9966 | 0.9937 | 0.9987 |
| **MAE (nM)** | 5 | 7.74 | 2.97 |
| **RMSE (nM)** | 7.77 | 10.61 | 4.72 |
<img width="1200" height="500" alt="Figure_C1" src="https://github.com/user-attachments/assets/794661d9-0ce5-49fb-aae4-e083cfccb45d" />

<img width="550" height="500" alt="Figure_C2" src="https://github.com/user-attachments/assets/9464afe4-b1ab-44ac-8660-bd43b631fe77" />

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#=========================
# 1. Load and reshape data
#=========================

concentrations = [5, 20, 50, 100, 150, 200, 250, 300, 350, 400]
all_data = []
all_labels = []

for conc in concentrations:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', f'SWV_curve_augmented_{conc}.csv')
    df = pd.read_csv(file_path)
    
    # Each file has 200 curves. Every 2 columns is one curve.
    for i in range(1, 201):
        # Extract the two columns for this specific curve
        v_col = f'Voltage_{i}'
        c_col = f'Current_{i}'
        
        # Convert to a 90x2 array
        curve_array = df[[v_col, c_col]].values 
        
        all_data.append(curve_array)
        all_labels.append(conc) # The target we want to predict

# Convert list to a 3D Numpy Array
X = np.array(all_data)   # Shape: (2000, 90, 2)
y = np.array(all_labels) # Concentration, shape: (2000,)

print(f"Data is now in 3D format: {X.shape}")

#=================================
# 2. Train, test, validation split
#=================================

# 2.1. Split into Train (85%) and Test (15%)
# 'stratify=y' ensures every concentration is represented in both sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2.2. Split the Train_full into Train (70% total) and Validation (15% total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
)

print(f"Train samples: {len(X_train)}")      # ~1400 curves
print(f"Validation samples: {len(X_val)}")   # ~300 curves
print(f"Test samples: {len(X_test)}")        # ~300 curves

# 2.3. NN fail if feed them raw current values (like 10^-6 A) alongside voltages (like 0.5 V), must scale them.
# Create the scaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Flatten to (Samples * Points, Features) to scale all values equally
X_train_reshaped = X_train.reshape(-1, 2)
X_val_reshaped = X_val.reshape(-1, 2)
X_test_reshaped = X_test.reshape(-1, 2)

# Fit on TRAIN only (to avoid data leakage) and transform others
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

print("Data is split and scaled. Ready for Neural Networks!")

#=================
# 3. Model A: 1D CNN
#=================
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

early_stoppingA = EarlyStopping(
    monitor='val_loss',      # Watch the validation loss
    patience=15,             # How many epochs to wait for improvement before stopping
    restore_best_weights=True # Crucial: returns the model to its best state, not the last one
)

#3.1. Build the 1D CNN model
def build_1d_cnn(input_shape):
    model = models.Sequential([
        # Layer 1: Extract local patterns (filters scan the 90 points)
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        # Layer 2: Deeper features
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # Flatten and transition to "Regression"
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), # Prevents overfitting
        
        # Output layer: 1 neuron for concentration (nM)
        layers.Dense(1, activation='linear') 
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Initialize the model using your 3D shape (90 points, 2 features)
input_shape = (90, 2)
cnn_model = build_1d_cnn(input_shape)

# 3.2. Train the model
history_cnn = cnn_model.fit(
    X_train_scaled, y_train_scaled, 
    epochs=500, 
    batch_size=32, 
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stoppingA],
    verbose=1
)

# 3.3. Get predictions
y_pred_cnn = cnn_model.predict(X_test_scaled).flatten()

#3.4. Calculate Metrics
y_pred_nM = scaler_y.inverse_transform(y_pred_cnn.reshape(-1, 1))
y_test_nM = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

r2_h = r2_score(y_test_scaled, y_pred_cnn)
mae_h = mean_absolute_error(y_test_nM, y_pred_nM)
rmse_h = np.sqrt(mean_squared_error(y_test_nM, y_pred_nM))

#================================
# 4. Performance metrics of model
#================================

print("--- Model A: 1D CNN Performance ---")
best_epochA = np.argmin(history_cnn.history['val_loss']) + 1
print("Best epoch:", best_epochA)
print(f"R-squared: {r2_h:.4f}")
print(f"MAE: {mae_h:.2f} nM")
print(f"RMSE: {rmse_h:.2f} nM")

#Best epoch: 49
#R-squared: 0.9966
#MAE: 5.00 nM
#RMSE: 7.77 nM

#=========
# 5. Plots
#=========

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE (Scaled)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

plot_history(history_cnn)

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    # Scatter plot
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Perfect prediction line
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    
    r2 = r2_score(y_true, y_pred)
    plt.title(f'Predicted vs. Actual Concentration\n$R^2 = {r2:.4f}$')
    plt.xlabel('Actual Concentration (nM)')
    plt.ylabel('Predicted Concentration (nM)')
    plt.grid(True, alpha=0.3)
    plt.show()

# y_test_nM and y_pred_nM from previous solution

plot_predictions(y_test_nM, y_pred_nM)

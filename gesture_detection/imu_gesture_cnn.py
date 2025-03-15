import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob
import matplotlib.pyplot as plt

# Load multiple IMU CSV files
def load_data(folder_path):
    all_files = glob.glob(folder_path + "/*.csv")
    dataframes = []
    labels = []
    
    for file in all_files:
        df = pd.read_csv(file)
        gesture_label = file.split("/")[-1].replace(".csv", "")  # Extract label from filename
        df["Gesture_Label"] = gesture_label  # Use filename as gesture label
        dataframes.append(df)
        labels.append(gesture_label)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df, labels

# Preprocess the data using time-based segmentation
def preprocess_data(df, window_size=100, step_size=50):
    X, y = [], []
    scaler = StandardScaler()
    df.iloc[:, 2:-1] = scaler.fit_transform(df.iloc[:, 2:-1])  # Normalize IMU features
    
    label_encoder = LabelEncoder()
    df['Gesture_Label'] = label_encoder.fit_transform(df['Gesture_Label'])  # Encode labels
    
    for gesture_label in df['Gesture_Label'].unique():
        subset = df[df['Gesture_Label'] == gesture_label]
        marked_indices = subset[subset['Annotation'] == 'marked'].index  # Select only marked regions
        
        for i in range(0, len(marked_indices) - window_size, step_size):
            window_indices = marked_indices[i:i+window_size]
            if len(window_indices) < window_size:
                continue  # Skip incomplete windows
            
            window = subset.loc[window_indices, subset.columns[2:-2]].values
            X.append(window)
            y.append(gesture_label)
    
    return np.array(X), np.array(y), label_encoder

# Define the 1D-CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and process training data
df, gesture_labels = load_data('training_data')  # Load from training_data folder
X, y, label_encoder = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = create_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(np.unique(y)))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and label encoder for deployment
model.save('gesture_model.h5')
np.save('label_encoder.npy', label_encoder.classes_)

# Function to test the model on new data
def test_model(test_file):
    df_test = pd.read_csv(f'testing_data/{test_file}')  # Load from testing_data folder
    scaler = StandardScaler()
    df_test.iloc[:, 2:-1] = scaler.fit_transform(df_test.iloc[:, 2:-1])
    
    model = load_model('gesture_model.h5')
    label_encoder = np.load('label_encoder.npy', allow_pickle=True)
    
    window_size = 100
    step_size = 50
    X_test, y_true, timestamps = [], [], []
    
    for i in range(0, len(df_test) - window_size, step_size):
        window = df_test.iloc[i:i+window_size, 2:-1].values
        X_test.append(window)
        y_true.append(df_test.iloc[i:i+window_size]['Annotation'].mode()[0])
        timestamps.append(df_test.iloc[i]['Time'])
    
    X_test = np.array(X_test)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['Time'], df_test.iloc[:, 2], label='IMU Signal')
    for i, ts in enumerate(timestamps):
        if y_pred[i] != -1:  # Highlight detected gestures
            plt.axvspan(ts, ts + window_size*0.01, color='red', alpha=0.3, label=f'Pred: {label_encoder[y_pred[i]]}')
    plt.legend()
    plt.title('Predicted Gestures on IMU Data')
    plt.show()
    
    # Ground Truth Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['Time'], df_test.iloc[:, 2], label='IMU Signal')
    for i, ts in enumerate(timestamps):
        if y_true[i] == 'marked':  # Highlight actual gestures
            plt.axvspan(ts, ts + window_size*0.01, color='green', alpha=0.3, label='Ground Truth')
    plt.legend()
    plt.title('Ground Truth Gestures on IMU Data')
    plt.show()
    
# Run test
test_model('test_model.csv')

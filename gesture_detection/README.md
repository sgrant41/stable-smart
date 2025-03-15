# IMU Gesture Recognition with 1D-CNN

This project uses a 1D Convolutional Neural Network (CNN) to recognize predefined gestures based on IMU (Inertial Measurement Unit) data. The model is trained to classify five specific gestures along with a non-gesture class and can run efficiently on edge devices.

## Project Structure

```
.
├── training_data/      # Folder containing CSV files for training
├── testing_data/       # Folder containing CSV files for testing
├── gesture_model.h5    # Saved trained model
├── label_encoder.npy   # Saved label encoder classes
├── imu_gesture_cnn.py  # Main script for training and testing
└── README.md           # Documentation
```

## Data Format

Each CSV file contains IMU sensor data with the following columns:

- `Time` (timestamp)
- `IMU_X`, `IMU_Y`, `IMU_Z` (accelerometer and gyroscope readings)
- `Annotation` ("marked" for gesture regions, "unmarked" otherwise)

Each CSV file is named after the corresponding gesture label (e.g., `kick.csv`, `wave.csv`). The sixth file contains random movements labeled as "none".

## Setup and Dependencies

Ensure you have the following installed:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## Training the Model

1. Place the training CSV files in the `training_data/` folder.
2. Run the following command to train the model:

```bash
python imu_gesture_cnn.py
```

3. The trained model (`gesture_model.h5`) and label encoder (`label_encoder.npy`) will be saved.

## Testing the Model

1. Place a test CSV file (e.g., `test_model.csv`) in the `testing_data/` folder.
2. Run the script to evaluate the model on new data:

```bash
python imu_gesture_cnn.py
```

3. The script will generate two plots:
   - **Predicted Gestures**: Shows detected gestures over IMU data.
   - **Ground Truth Gestures**: Highlights actual gestures based on annotations.

## Notes

- The model uses a sliding window approach for segmentation.
- IMU features are normalized before training.
- Ensure the CSV files are formatted correctly for best results.

## Future Improvements

- Optimize model for mobile deployment.
- Explore other deep learning architectures like LSTMs or Transformers.
- Collect more diverse data to improve accuracy across different users.


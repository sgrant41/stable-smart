import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from collections import defaultdict

DATA_FOLDER = "imu_data"

def get_latest_csv():
    """Finds the most recent CSV file in the imu_data folder."""
    files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("imu_data_") and f.endswith(".csv")]
    if not files:
        print("No IMU data files found.")
        return None
    files.sort(reverse=True)  # Latest file first
    return os.path.join(DATA_FOLDER, files[0])

# Get the latest CSV file or let the user choose a file
latest_csv = get_latest_csv()
if latest_csv:
    print(f"Found latest file: {latest_csv}")
    choice = input("Press Enter to use this file or type another filename: ").strip()
    if choice:
        CSV_FILENAME = os.path.join(DATA_FOLDER, choice)
    else:
        CSV_FILENAME = latest_csv
else:
    CSV_FILENAME = input("Enter the CSV filename to plot (inside imu_data/): ").strip()
    CSV_FILENAME = os.path.join(DATA_FOLDER, CSV_FILENAME)

# Read CSV data and group by device
data_by_device = defaultdict(lambda: {"time": [], "accel_x": [], "accel_y": [], "accel_z": [],
                                       "gyro_x": [], "gyro_y": [], "gyro_z": [], "annotation": []})
all_annotation_times = []

try:
    with open(CSV_FILENAME, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        for row in reader:
            device = row[0]
            t = float(row[1])
            ax = float(row[2])
            ay = float(row[3])
            az = float(row[4])
            gx = float(row[5])
            gy = float(row[6])
            gz = float(row[7])
            annot = row[8]
            data_by_device[device]["time"].append(t)
            data_by_device[device]["accel_x"].append(ax)
            data_by_device[device]["accel_y"].append(ay)
            data_by_device[device]["accel_z"].append(az)
            data_by_device[device]["gyro_x"].append(gx)
            data_by_device[device]["gyro_y"].append(gy)
            data_by_device[device]["gyro_z"].append(gz)
            is_annotated = (annot == "Annotated")
            data_by_device[device]["annotation"].append(is_annotated)
            if is_annotated:
                all_annotation_times.append(t)
    
    # Compute the union of annotation segments (group times that are close together)
    all_annotation_times.sort()
    annotated_segments = []
    if all_annotation_times:
        seg_start = all_annotation_times[0]
        seg_end = all_annotation_times[0]
        threshold = 0.5  # seconds; adjust if needed
        for t in all_annotation_times[1:]:
            if t - seg_end <= threshold:
                seg_end = t
            else:
                annotated_segments.append((seg_start, seg_end))
                seg_start = t
                seg_end = t
        annotated_segments.append((seg_start, seg_end))
    
    # Create two subplots: one for acceleration and one for gyroscope data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Get a color cycle for devices
    colors = plt.cm.tab10.colors
    device_names = list(data_by_device.keys())
    
    for idx, device in enumerate(device_names):
        color = colors[idx % len(colors)]
        dev_data = data_by_device[device]
        t_array = np.array(dev_data["time"])
        
        # Plot acceleration channels for this device
        ax1.plot(t_array, dev_data["accel_x"], label=f'{device} Accel X', color=color, linestyle='-')
        ax1.plot(t_array, dev_data["accel_y"], label=f'{device} Accel Y', color=color, linestyle='--')
        ax1.plot(t_array, dev_data["accel_z"], label=f'{device} Accel Z', color=color, linestyle=':')
        
        # Plot gyroscope channels for this device
        ax2.plot(t_array, dev_data["gyro_x"], label=f'{device} Gyro X', color=color, linestyle='-')
        ax2.plot(t_array, dev_data["gyro_y"], label=f'{device} Gyro Y', color=color, linestyle='--')
        ax2.plot(t_array, dev_data["gyro_z"], label=f'{device} Gyro Z', color=color, linestyle=':')
    
    # Highlight annotated regions on both subplots
    for seg_start, seg_end in annotated_segments:
        ax1.axvspan(seg_start, seg_end, color='red', alpha=0.3)
        ax2.axvspan(seg_start, seg_end, color='red', alpha=0.3)
    
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.set_title("IMU Acceleration Data by Device")
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Gyroscope (°/s)")
    ax2.set_title("IMU Gyroscope Data by Device")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILENAME}' was not found.")
except Exception as e:
    print(f"Error reading file: {e}")
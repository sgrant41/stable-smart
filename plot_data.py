import matplotlib.pyplot as plt
import csv
import os

DATA_FOLDER = "imu_data"  # Folder where CSV files are stored

def get_latest_csv():
    """Find the most recent IMU CSV file in the directory."""
    files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("imu_data_") and f.endswith(".csv")]
    if not files:
        print("No IMU data files found.")
        return None
    files.sort(reverse=True)  # Sort by latest timestamp
    return os.path.join(DATA_FOLDER, files[0])

# Automatically get the latest CSV file, or prompt the user for a specific one
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

# Read data from CSV
time_vals, accel_x, accel_y, accel_z = [], [], [], []
gyro_x, gyro_y, gyro_z = [], [], []

try:
    with open(CSV_FILENAME, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            time_vals.append(float(row[0]))
            accel_x.append(float(row[1]))
            accel_y.append(float(row[2]))
            accel_z.append(float(row[3]))
            gyro_x.append(float(row[4]))
            gyro_y.append(float(row[5]))
            gyro_z.append(float(row[6]))

    # Plot Acceleration Data
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time_vals, accel_x, label='Accel X', color='r')
    plt.plot(time_vals, accel_y, label='Accel Y', color='g')
    plt.plot(time_vals, accel_z, label='Accel Z', color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("IMU Acceleration Data")
    plt.legend()
    plt.grid(True)

    # Plot Gyroscope Data
    plt.subplot(2, 1, 2)
    plt.plot(time_vals, gyro_x, label='Gyro X', color='c')
    plt.plot(time_vals, gyro_y, label='Gyro Y', color='m')
    plt.plot(time_vals, gyro_z, label='Gyro Z', color='y')
    plt.xlabel("Time (s)")
    plt.ylabel("Gyroscope (°/s)")
    plt.title("IMU Gyroscope Data")
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{CSV_FILENAME}' was not found.")
except Exception as e:
    print(f"Error reading file: {e}")
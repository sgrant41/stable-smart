import matplotlib.pyplot as plt
import csv

CSV_FILENAME = "imu_data.csv"

# Read data from CSV
time_vals, accel_x, accel_y, accel_z = [], [], [], []
gyro_x, gyro_y, gyro_z = [], [], []

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

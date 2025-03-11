import matplotlib.pyplot as plt
import csv

CSV_FILENAME = "imu_data.csv"

# Read data from CSV
time_vals, accel_x, accel_y, accel_z = [], [], [], []

with open(CSV_FILENAME, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        time_vals.append(float(row[0]))
        accel_x.append(float(row[1]))
        accel_y.append(float(row[2]))
        accel_z.append(float(row[3]))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time_vals, accel_x, label='Accel X', color='r')
plt.plot(time_vals, accel_y, label='Accel Y', color='g')
plt.plot(time_vals, accel_z, label='Accel Z', color='b')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("IMU Acceleration Data Over Time")
plt.legend()
plt.grid(True)
plt.show()

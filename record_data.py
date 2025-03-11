import asyncio
import csv
import time
import os
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notification characteristic

# CSV file setup
CSV_FILENAME = "imu_data.csv"
FIELDNAMES = ["Time (s)", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]

# Create the CSV file if it doesn't exist and write the header
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FIELDNAMES)  # Write header

start_time = None

def notification_handler(sender, received_data):
    """Callback for handling incoming BLE notifications."""
    global start_time
    try:
        line = received_data.decode('utf-8').strip()
        print("Received:", line)  # Debugging
        parts = line.split(',')
        if len(parts) >= 7:  # Ensure enough data points
            timestamp = float(parts[0])
            ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])

            if start_time is None:
                start_time = time.time()

            elapsed_time = time.time() - start_time

            # Save to CSV file immediately
            with open(CSV_FILENAME, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([elapsed_time, ax, ay, az, gx, gy, gz])
                
    except Exception as e:
        print("Error decoding notification data:", e)

async def run_ble():
    """Scan for the BLE device, connect, and subscribe to notifications."""
    print("Scanning for BLE device...")
    devices = await BleakScanner.discover(timeout=5.0)
    target = next((d for d in devices if d.name == "M5-IMU"), None)

    if target is None:
        print("Target BLE device not found.")
        return

    async with BleakClient(target) as client:
        print("Connected to", target.address)
        await client.start_notify(UART_TX_CHAR_UUID, notification_handler)
        
        try:
            while True:
                await asyncio.sleep(1.0)  # Keep connection alive
        except KeyboardInterrupt:
            print("\nRecording stopped by user. Data saved.")
            exit(0)

async def main():
    await run_ble()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram exited.")

import asyncio
import csv
import time
import os
from datetime import datetime
import keyboard  # For detecting key presses; ensure it's installed (pip install keyboard)
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Folder to store CSV files
DATA_FOLDER = "imu_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Generate a timestamped filename inside imu_data/
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_FILENAME = os.path.join(DATA_FOLDER, f"imu_data_{timestamp}.csv")
FIELDNAMES = ["Device", "Time (s)", "Accel X", "Accel Y", "Accel Z",
              "Gyro X", "Gyro Y", "Gyro Z", "Annotation"]

# Create and initialize the CSV file with header
with open(CSV_FILENAME, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(FIELDNAMES)

# Global start time for all devices (set in main)
global_start_time = None

def make_notification_handler(device_name):
    """Creates a notification handler that tags data with the device name."""
    def notification_handler(sender, received_data):
        global global_start_time
        try:
            line = received_data.decode('utf-8').strip()
            print(f"Received from {device_name}: {line}")
            parts = line.split(',')
            if len(parts) >= 7:
                # Parse sensor values; note that parts[0] may be a device timestamp (ignored here)
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
                
                # Use a common start time for elapsed time computation
                if global_start_time is None:
                    global_start_time = time.time()
                elapsed_time = time.time() - global_start_time

                # Check if the space key is held down for annotation
                annotation = "Annotated" if keyboard.is_pressed('space') else "Unmarked"
                
                # Append the data row to the CSV file
                with open(CSV_FILENAME, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([device_name, elapsed_time, ax, ay, az, gx, gy, gz, annotation])
        except Exception as e:
            print(f"Error decoding data from {device_name}: {e}")
    return notification_handler

async def run_ble_for_device(device):
    """Connects to a given device and starts notifications."""
    try:
        async with BleakClient(device) as client:
            print(f"Connected to {device.name} at {device.address}")
            handler = make_notification_handler(device.name)
            await client.start_notify(UART_TX_CHAR_UUID, handler)
            # Keep the connection alive
            while True:
                await asyncio.sleep(1.0)
    except Exception as e:
        print(f"Connection error with {device.name}: {e}")

async def main():
    global global_start_time
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)
    # Select devices with names starting with "M5-IMU"
    target_devices = [d for d in devices if d.name and d.name.startswith("M5-IMU")]
    
    if not target_devices:
        print("No M5-IMU devices found.")
        return

    print("Found devices:")
    for d in target_devices:
        print(f" - {d.name} ({d.address})")
    
    # Set a common start time for recording
    global_start_time = time.time()
    
    # Create and run a task for each discovered device
    tasks = [asyncio.create_task(run_ble_for_device(device)) for device in target_devices]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\nRecording stopped. Data saved to {CSV_FILENAME}.")
        for task in tasks:
            task.cancel()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram exited.")

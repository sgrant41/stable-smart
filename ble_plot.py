import asyncio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notification characteristic

# Deques to store data (max 100 samples)
maxlen = 100
times = deque(maxlen=maxlen)
accel_x = deque(maxlen=maxlen)
accel_y = deque(maxlen=maxlen)
accel_z = deque(maxlen=maxlen)

def notification_handler(sender, data):
    """Callback for handling incoming BLE notifications."""
    try:
        # Decode CSV string: timestamp, ax, ay, az, gx, gy, gz
        line = data.decode('utf-8').strip()
        print("Received:", line)  # Debug: verify that data is arriving
        parts = line.split(',')
        if len(parts) >= 4:
            t = float(parts[0])
            ax_val = float(parts[1])
            ay_val = float(parts[2])
            az_val = float(parts[3])
            times.append(t)
            accel_x.append(ax_val)
            accel_y.append(ay_val)
            accel_z.append(az_val)
    except Exception as e:
        print("Error decoding notification data:", e)

async def run_ble():
    """Scan for the BLE device, connect, and subscribe to notifications."""
    print("Scanning for BLE device...")
    devices = await BleakScanner.discover(timeout=5.0)
    target = None
    for d in devices:
        # Ensure the device name exactly matches your peripheral's advertised name.
        print(d.name)
        if d.name == "M5-IMU":
            target = d
            break
    if target is None:
        print("Target BLE device not found.")
        return
    async with BleakClient(target) as client:
        print("Connected to", target.address)
        await client.start_notify(UART_TX_CHAR_UUID, notification_handler)
        # Keep the connection alive.
        while True:
            await asyncio.sleep(1.0)

# Set up Matplotlib for live plotting.
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Accel X')
line2, = ax.plot([], [], label='Accel Y')
line3, = ax.plot([], [], label='Accel Z')
ax.legend()
ax.set_xlabel("Sample")
ax.set_ylabel("Acceleration (m/s²)")
ax.set_title("Live IMU Acceleration Data")

def init_plot():
    ax.set_xlim(0, maxlen)
    ax.set_ylim(-10, 10)  # Adjust limits based on your sensor's output.
    return line1, line2, line3

def update_plot(frame):
    # Update lines with new data from deques.
    line1.set_data(range(len(accel_x)), list(accel_x))
    line2.set_data(range(len(accel_y)), list(accel_y))
    line3.set_data(range(len(accel_z)), list(accel_z))
    return line1, line2, line3

# Set up FuncAnimation (this will update the plot regularly)
ani = animation.FuncAnimation(
    fig,
    update_plot,
    init_func=init_plot,
    interval=100,
    blit=True,
    cache_frame_data=False
)

async def main():
    # Start BLE communication as a background asyncio task.
    ble_task = asyncio.create_task(run_ble())
    # Use non-blocking interactive mode for Matplotlib.
    plt.show(block=False)
    # Run a loop to keep updating the plot.
    while True:
        plt.pause(0.1)  # Allow Matplotlib to process GUI events.
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(main())

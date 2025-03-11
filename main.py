# M5 Device BLE UART Server with IMU Data Streaming

import ubluetooth
import time

from machine import I2C, Pin, PWM
i2c = I2C(0, scl=Pin(22), sda=Pin(21)) #I2C comms

p4 = Pin(4, Pin.OUT) # set GPIO4 as an output pin, this'll control power
p2 = Pin(2, Pin.OUT) # GPIO2 is buzzer
p19 = Pin(19, Pin.OUT) #Red LED/IR transmitter




# Import IMU sensor driver
try:
    from mpu6886 import MPU6886  # For M5 devices with an MPU6886 sensor (like this one)
except ImportError:
    print("MPU6886 library not found. Please install the required driver.")
    MPU6886 = None

class BLEUART:
    def __init__(self, ble, name="M5-IMU"):
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        self._connections = set()
        self._rx_buffer = bytearray()

        # Define the Nordic UART Service (NUS) UUIDs.
        UART_UUID      = ubluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
        UART_TX_UUID   = ubluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")  # notify
        UART_RX_UUID   = ubluetooth.UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E")  # write

        self._UART_TX = (UART_TX_UUID, ubluetooth.FLAG_NOTIFY)
        self._UART_RX = (UART_RX_UUID, ubluetooth.FLAG_WRITE)
        self._UART_SERVICE = (UART_UUID, (self._UART_TX, self._UART_RX))
        ((self._tx_handle, self._rx_handle),) = self._ble.gatts_register_services((self._UART_SERVICE,))
        
        self._advertise(name)

    def _irq(self, event, data):
        # Handling various BLE events.
        if event == 1:  # _IRQ_CENTRAL_CONNECT
            conn_handle, addr_type, addr = data
            self._connections.add(conn_handle)
        elif event == 2:  # _IRQ_CENTRAL_DISCONNECT
            conn_handle, addr_type, addr = data
            self._connections.remove(conn_handle)
            self._advertise()  # Restart advertising on disconnect.
        elif event == 3:  # _IRQ_GATTS_WRITE
            conn_handle, value_handle = data
            if value_handle == self._rx_handle:
                self._rx_buffer += self._ble.gatts_read(self._rx_handle)

    def send(self, data):
        # Send data to all connected devices.
        for conn_handle in self._connections:
            self._ble.gatts_notify(conn_handle, self._tx_handle, data)

    def _advertise(self, name="M5-IMU"):
        # Use a bytes literal for the flags.
        adv_payload = bytearray(b'\x02\x01\x06') + bytearray((len(name) + 1, 0x09)) + name.encode()
        self._ble.gap_advertise(100, adv_payload)

# ============================================ Actual Code Here =========================================

# Initialize battery/power supply
p4.on() #set pin 4 to high

# Initialize power indicator
pwmred = PWM(p19) #make PWM object for the red LED (GPIO19), this should also turn it on
pwmred.freq(1000) #frequency of 1 kHz
pwmred.duty(5) #apply duty cycle


# Initialize BLE
ble = ubluetooth.BLE()
ble_uart = BLEUART(ble, name="Tony-IMU") #may need to change the naming syntax with multiple devices together

# Initialize the IMU sensor
if MPU6886:
    imu = MPU6886(i2c)
else:
    imu = None

while True:
    #buzzer.value(1)
    if imu:
        # Read IMU sensor data: assume functions return (x, y, z)
        accel = imu.acceleration()   # e.g., (ax, ay, az)
        gyro  = imu.gyro()     # e.g., (gx, gy, gz)
    else:
        # Dummy data if sensor is not available
        accel = (0, 0, 0)
        gyro  = (0, 0, 0)
    
    # Create CSV string: timestamp, ax, ay, az, gx, gy, gz
    data_str = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(
        time.ticks_ms() / 1000,
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2]
    )
    
    # Send the CSV string over BLE (as bytes)
    ble_uart.send(data_str.encode('utf-8'))
    
    # Wait 100ms before the next reading
    time.sleep(0.1)

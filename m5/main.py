import ubluetooth
import time

from machine import I2C, Pin, PWM, ADC
i2c = I2C(0, scl=Pin(22), sda=Pin(21)) #initialize I2C comms

#Initialize Pins
p4 = Pin(4, Pin.OUT) # set GPIO4 as an output pin, this'll control power
p19 = Pin(19, Pin.OUT) #Red LED/IR transmitter
Abutt = Pin(37, Pin.IN, Pin.PULL_UP) #Button A (the one on the same face as the screen)
adc = ADC(Pin(35)) #ADC connected to onboard battery

#Initialize Variables
NAME = "M5-IMU-Name" #device name, used in BLE
IMU_buffer = [] #blank buffer to hold multiple IMU readings

packet_size = 1 #how many readings to send at once
    #^^^^^^^^^^^^ This should stay at 1 for now until laptop code is updated to receive packets of multiple sensor readings
IMU_delay = 100 #time to wait between IMU readings
battery_delay = 1000 #time between battery checks

batt_max = 3.7 #max battery voltage, should probably actually measure this

# Import IMU sensor driver
try:
    from mpu6886 import MPU6886
except ImportError:
    print("MPU6886 library not found. Make sure the driver (mpu6886.py) is installed.")
    MPU6886 = None

#======================================  Defines ===========================================
class BLEUART:
    def __init__(self, ble, name=NAME):
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

    def _advertise(self, name=NAME):
        # Use a bytes literal for the flags.
        adv_payload = bytearray(b'\x02\x01\x06') + bytearray((len(name) + 1, 0x09)) + name.encode()
        self._ble.gap_advertise(100, adv_payload) #advertising interval (currently 100 ms)
#End BLE defs \(.o.)/
        
def read_imu():
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
        time.ticks_ms() / 1000, #time in seconds
        accel[0], accel[1], accel[2],
        gyro[0], gyro[1], gyro[2]
    )
    return data_str

def transmit_data(buffer):
    if len(buffer) >= packet_size:
        packet = "\n".join(buffer) #merge entries of buffer, separate with new lines
        ble_uart.send(packet.encode('utf-8')) #send packet
        buffer.clear() #erase buffer

def read_batt_volt(): #returns current voltage of the battery
    reading = adc.read()
    voltage = reading * (batt_max / 4095)
    return voltage

def read_batt_pct():
    batt_now = read_batt_volt() #current voltage
    pct = (batt_now / batt_init) * 100
    return pct

def guess_rem_life():
    batt_now = read_batt_volt()
    runtime_ms = time.ticks_diff(current_time,init_time) #how much time (ms) has elapsed since start
    runtime_V = batt_init - batt_now #how much voltage was drawn since start
    dV_dt = runtime_ms / runtime_V
    t_rem_ms = batt_now / dV_dt #number of milliseconds of life left if power draw doesn't change
    t_rem_s = t_rem_ms / 1000
    return t_rem_s
    
    
#==================================== Component Initialization ===================================================
# Initialize battery/power supply
p4.on() #set pin 4 to high
batt_init = read_batt_volt() #battery voltage upon powering on

#initialize timers
init_time = time.ticks_ms()
current_time = init_time
last_IMU = current_time
last_battery = current_time

# Initialize power indicator LED
pwmred = PWM(p19, freq=500) #make PWM object for the red LED (GPIO19)
pwmred.duty(0) #turn off LED

# Create interrupt for button A to turn on power indicator
def indic_handler(pin):
    pwmred.duty(40)
    time.sleep(0.5) #leave on for n seconds
    pwmred.duty(0) #turn led off again
Abutt.irq(trigger=Pin.IRQ_FALLING, handler=indic_handler) #assign interrupt handler to button A

# Initialize BLE
ble = ubluetooth.BLE()
ble_uart = BLEUART(ble)

# Initialize the IMU
if MPU6886:
    imu = MPU6886(i2c)
else:
    imu = None

#========================================== Superloop ==================================================
while True:
    current_time = time.ticks_ms()
    
    if time.ticks_diff(current_time, last_IMU) >= IMU_delay: 
        imu_str = read_imu()
        IMU_buffer.append(imu_str) #add reading to buffer
        last_IMU = current_time #reset IMU timer

    transmit_data(IMU_buffer) #send a packet once enough readings are collected
    
    if time.ticks_diff(current_time, last_battery) >= battery_delay:
        percent = read_batt_pct()
        print("{:.2f}% Remaining".format(percent)) #for debug, not essential
        battery_delay = current_time #reset battery timer

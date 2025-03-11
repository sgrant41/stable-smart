# stable-smart
File sharing for Stable Smart capstone (the horse fitbit one).

## Current Files
On the board: main.py and mpu6886.py
On your computer: ble_plot.py

## Setup
### MicroPython Install
If you're starting with a fresh M5Stick you'll need to install MicroPython. You'll need to download [this firmware version](https://micropython.org/download/ESP32_GENERIC/) and [Thonny](https://thonny.org/).
1. Install Thonny and open a window. In the top toolbar, navigate to Tools>Options>Interpreter. Change the first dropdown to Micropython(ESP32).
2. Connect the M5Stick and select the relevant COM port from the second dropdown.
3. Click the blue text in the bottom right of the window ("Install or update MicroPython (esptool)")
4. In the new window, click the hamburger button to the left of Install, click "Select local Micropython image"
5. Select the firmware .bin from the dialog. Once this is done, the remaining fields should autofill.
6. Make sure "Erase all flash..." is checked and click Install. The process should take about 5 minutes.
MicroPython should now be installed on the M5, make sure to mark it in the MCU Inventory spreadsheet.

### Uploading Code
1. In the main window of Thonny, create a new file and paste the contents of main.py
2. Scroll down to the initialization lines near the end and edit the name of ble_uart to reflect the device used (ex. "M5-Tony")
3. Save As > MicroPython device > type "main.py" and save
4. Repeat for mpu6886.py, omitting the name editing part

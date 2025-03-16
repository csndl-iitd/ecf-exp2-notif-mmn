import serial

import serial.tools.list_ports
from psychopy import sound, gui, visual, core


# List available ports

#print(serial.tools.list_ports.comports()[0])



# Assuming COM4 is the correct port

try:

    ser = serial.Serial('COM4', baudrate=115200)  # Adjust timeout if needed
    print("Serial port opened successfully")
    ser.write(bytes(255))
    #core.wait(10)
    print("Port close")
    ser.close()

except serial.SerialException as e:

    print("Error opening serial port:", e)

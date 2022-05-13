import RPi.GPIO as GPIO

import time

Pins = [29,31,33,35,37]
def setup():
    GPIO.setmode(GPIO.BOARD)
    for pin in Pins:
        GPIO.setup(pin, GPIO.OUT,initial=GPIO.LOW)
    GPIO.setwarnings(False)
    print("Setup Done!!!")

def On_LED(Led_Pin):
    GPIO.output(Led_Pin, GPIO.HIGH)

def Off_LED(Led_Pin):
    GPIO.output(Led_Pin, GPIO.LOW)

def Off_All():
    for pin in Pins:
        Off_LED(pin)

def On_All():
    for pin in Pins:
        On_LED(pin)    

def Clean():
    Off_All()
    GPIO.cleanup()

if __name__ == '__main__':
    setup()
    try:
        while True:
            # Off_All()
            # time.sleep(1)
            On_All()
    finally:    
        Clean()
        print("Done")


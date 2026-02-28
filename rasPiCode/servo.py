from gpiozero import Servo
from time import sleep

# Mapping your nanosecond/microsecond specs to seconds
# 500us = 0.0005s, 2500us = 0.0025s
servo = Servo(4, min_pulse_width=0.0005, max_pulse_width=0.0025)

try:
    print("Testing continuous servo with precise pulse timing...")
    
    # Spin one direction
    servo.min()
    print("Spinning direction A")
    sleep(2)
    
    # Neutral/Stop
    servo.mid()
    print("Neutral/Stop")
    sleep(1)
    
    # Spin other direction
    servo.max()
    print("Spinning direction B")
    sleep(2)
    
    # Neutral/Stop
    servo.mid()
    print("Stopping")

except KeyboardInterrupt:
    servo.detach() # Safely stop sending PWM signals
    print("\nProgram stopped")
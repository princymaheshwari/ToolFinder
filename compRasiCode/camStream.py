import socket
import cv2
import struct
import numpy as np

# Change to the actual IP address of your Raspberry Pi
PI_IP = '192.168.50.232' 
PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((PI_IP, PORT))

data = b""
payload_size = struct.calcsize(">L")

try:
    while True:
        # Retrieve message size
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        
        # Retrieve actual image data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        # Decode and show
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow('Robotics Stream', frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    client_socket.close()
    cv2.destroyAllWindows()
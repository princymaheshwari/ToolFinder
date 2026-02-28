import cv2
import socket
import struct

# Initialize socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

print("Waiting for connection on port 9999...")
client_socket, addr = server_socket.accept()
print(f'Connected to: {addr}')

cap = cv2.VideoCapture(4) # Adjust if your camera index is different

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Encode frame as JPEG
        _, jpg_data = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        data = jpg_data.tobytes()
        
        # Pack the length of the data (4 bytes) + the data itself
        # This prevents the receiver from getting "mixed up" packets
        message = struct.pack(">L", len(data)) + data
        client_socket.sendall(message)
finally:
    client_socket.close()
    cap.release()
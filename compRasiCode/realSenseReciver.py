import socket
import struct
import numpy as np
import cv2
import json

PI_IP = '192.168.50.232'
PORT  = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((PI_IP, PORT))
print("Connected.")

def recvall(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def recv_bytes(sock) -> bytes:
    size = struct.unpack(">L", recvall(sock, 4))[0]
    return recvall(sock, size)

MAX_DEPTH_MM = 4000

try:
    while True:
        color_data = recv_bytes(client_socket)
        depth_data = recv_bytes(client_socket)
        imu_data   = recv_bytes(client_socket)

        # Decode color
        color_frame = cv2.imdecode(np.frombuffer(color_data, np.uint8), cv2.IMREAD_COLOR)

        # Decode + colorize depth
        depth_raw = cv2.imdecode(np.frombuffer(depth_data, np.uint8), cv2.IMREAD_UNCHANGED)
        depth_vis = (np.clip(depth_raw, 0, MAX_DEPTH_MM).astype(np.float32) / MAX_DEPTH_MM * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Decode IMU
        imu = json.loads(imu_data.decode())
        accel, gyro = imu["accel"], imu["gyro"]

        # Overlay IMU on color frame
        def put(img, text, row):
            cv2.putText(img, text, (10, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        put(color_frame, f"Accel  x:{accel['x']:+.2f}  y:{accel['y']:+.2f}  z:{accel['z']:+.2f} m/s²", 20)
        put(color_frame, f"Gyro   x:{gyro['x']:+.2f}   y:{gyro['y']:+.2f}   z:{gyro['z']:+.2f} rad/s", 42)

        cv2.imshow('RGB', color_frame)
        cv2.imshow('Depth', depth_colored)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    client_socket.close()
    cv2.destroyAllWindows()
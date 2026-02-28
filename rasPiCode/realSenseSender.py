import socket
import struct
import numpy as np
import pyrealsense2 as rs
import cv2
import json
import threading

# ── IMU pipeline (separate from video) ─────────────────────────────
imu_pipeline = rs.pipeline()
imu_config = rs.config()
imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
imu_config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
imu_pipeline.start(imu_config)

# ── Video pipeline ──────────────────────────────────────────────────
video_pipeline = rs.pipeline()
video_config = rs.config()
video_config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
video_config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
video_pipeline.start(video_config)

# ── Shared IMU state (updated by background thread) ────────────────
imu_lock = threading.Lock()
latest_imu = {
    "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
    "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
}

def imu_reader():
    while True:
        try:
            frames = imu_pipeline.wait_for_frames()
            for frame in frames:
                t = frame.profile.stream_type()
                d = frame.as_motion_frame().get_motion_data()
                val = {"x": d.x, "y": d.y, "z": d.z}
                with imu_lock:
                    if t == rs.stream.accel:
                        latest_imu["accel"] = val
                    elif t == rs.stream.gyro:
                        latest_imu["gyro"] = val
        except Exception:
            break

imu_thread = threading.Thread(target=imu_reader, daemon=True)
imu_thread.start()

# ── Socket setup ────────────────────────────────────────────────────
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

print("Waiting for connection on port 9999...")
client_socket, addr = server_socket.accept()
print(f'Connected to: {addr}')

def send_bytes(sock, data: bytes):
    sock.sendall(struct.pack(">L", len(data)) + data)

try:
    while True:
        frames = video_pipeline.wait_for_frames()

        # Color
        color_img = np.asanyarray(frames.get_color_frame().get_data())
        _, color_jpg = cv2.imencode('.jpg', color_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Depth
        depth_arr = np.asanyarray(frames.get_depth_frame().get_data())  # uint16 mm
        _, depth_png = cv2.imencode('.png', depth_arr)

        # IMU snapshot
        with imu_lock:
            imu_snapshot = json.dumps(latest_imu).encode()

        send_bytes(client_socket, color_jpg.tobytes())
        send_bytes(client_socket, depth_png.tobytes())
        send_bytes(client_socket, imu_snapshot)

except (BrokenPipeError, ConnectionResetError):
    print("Client disconnected.")
finally:
    client_socket.close()
    server_socket.close()
    video_pipeline.stop()
    imu_pipeline.stop()
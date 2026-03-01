import asyncio
import websockets
import json
import requests
from kinematics import solve  # Your existing solver

ESP32_IP = "192.168.1.100"

async def bridge_handler():
    # Connects to your backend's websocket
    uri = "ws://localhost:9998"
    async with websockets.connect(uri) as websocket:
        print("Connected to backend...")
        async for message in websocket:
            data = json.loads(message)
            
            # Expecting message format: {"type": "target", "x": 80.0, "y": 40.0}
            if data.get('type') == 'target':
                x, y = data['x'], data['y']
                
                # Perform math on the client (the "Brain")
                yaw, pitch = solve(x, y)
                
                # Send simple command to ESP32 (the "Muscle")
                try:
                    requests.get(f"http://{ESP32_IP}/move?yaw={yaw:.2f}&pitch={pitch:.2f}")
                    print(f"Sent: Yaw={yaw:.2f}, Pitch={pitch:.2f}")
                except Exception as e:
                    print(f"ESP32 unreachable: {e}")

if __name__ == "__main__":
    asyncio.run(bridge_handler())
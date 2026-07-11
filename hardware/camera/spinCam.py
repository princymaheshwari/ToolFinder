import PySpin
import cv2
import asyncio
import websockets
import time
import threading
import numpy as np
from pypylon import pylon

# Global thread-safe frames
flir_frame = None
webcam_frame = None
pylon_frame = None

def capture_flir():
    global flir_frame
    processor = PySpin.ImageProcessor()
    # Ensure we use high-quality color processing
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
    
    while True:
        system = PySpin.System.GetInstance()
        try:
            cam_list = system.GetCameras()
            if cam_list.GetSize() == 0:
                raise Exception("No FLIR camera found")
            
            cam = cam_list[0]
            cam.Init()
            
            # --- COLOR ENFORCEMENT BLOCK ---
            nodemap = cam.GetNodeMap()
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            
            # Try to set a Bayer format (common for color cameras)
            # We iterate through potential Bayer formats if one fails
            bayer_formats = ["BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"]
            for fmt in bayer_formats:
                entry = node_pixel_format.GetEntryByName(fmt)
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    node_pixel_format.SetIntValue(entry.GetValue())
                    print(f"FLIR: Pixel format set to {fmt}")
                    break
            # -------------------------------

            cam.BeginAcquisition()
            print("FLIR: Acquisition started.")
            
            while True:
                image_result = cam.GetNextImage(1000)
                if not image_result.IsIncomplete():
                    # The processor now knows to convert the specific Bayer format above
                    converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                    flir_frame = converted.GetNDArray().copy()
                image_result.Release()
        except Exception as e:
            print(f"FLIR Error: {e}. Retrying in 5s...")
            flir_frame = None
            time.sleep(5)
        finally:
            if 'cam' in locals():
                try: 
                    if cam.IsStreaming(): cam.EndAcquisition()
                    cam.DeInit()
                except: pass
                del cam
            system.ReleaseInstance()

def capture_normal():
    global webcam_frame
    while True:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            time.sleep(5)
            continue
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            webcam_frame = frame
        
        webcam_frame = None
        cap.release()

def capture_pylon():
    global pylon_frame
    while True:
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            device = tl_factory.CreateFirstDevice()
            camera = pylon.InstantCamera(device)
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            while camera.IsGrabbing():
                grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    # Ensure format is BGR for compatibility
                    pylon_frame = cv2.cvtColor(grab.Array, cv2.COLOR_BayerRG2BGR)
                grab.Release()
        except Exception as e:
            print(f"Pylon Error: {e}. Retrying...")
            pylon_frame = None
            time.sleep(5)

async def stream_handler(websocket, frame_type):
    while True:
        if frame_type == "flir": frame = flir_frame
        elif frame_type == "webcam": frame = webcam_frame
        else: frame = pylon_frame
        
        if frame is not None:
            # Downsample and encode
            small = cv2.resize(frame, (640, 480))
            _, jpeg = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            await websocket.send(jpeg.tobytes())
        await asyncio.sleep(0.03)

async def main():
    # Start hardware threads
    threading.Thread(target=capture_flir, daemon=True).start()
    threading.Thread(target=capture_normal, daemon=True).start()
    threading.Thread(target=capture_pylon, daemon=True).start()
    
    # Start servers
    async with websockets.serve(lambda ws: stream_handler(ws, "flir"), "0.0.0.0", 9999), \
               websockets.serve(lambda ws: stream_handler(ws, "webcam"), "0.0.0.0", 10000), \
               websockets.serve(lambda ws: stream_handler(ws, "pylon"), "0.0.0.0", 10001):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
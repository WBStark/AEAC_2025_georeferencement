from dronekit import Command, connect, VehicleMode, mavutil
from localisation_feux import localisationFeux
import cv2
import time
import serial
import os
from ultralytics import YOLO

camId = 0
maxRetries = 100

camFOV = 55 #in degrees # NEED TO CHANGE IF  NEEDED ?
camRes = [1080, 720] #x ,y # NEED TO CHANGE IF  NEEDED ?

# yolo and telemetry
model = YOLO("yolov8n.pt") # use our trained model
image_path = "" # Useless cause we<re gonna use opencv to read the camera feed
yolo_txt_output_dir = ""
telemetry_output_dir = ""
the_mission_is_done_and_all_txt_files_are_saved = 1

gimbal_pitch = 0
gimbal_roll = 0
gimbal_yaw = 0



class Coordinate:
    lat: float
    lon: float
    alt: float

def debPrint(message:str):
    global pargs
    if pargs.debugprints:
        print(message)

def main():
    debPrint("Starting AutoLander")

    #setup camera
    debPrint("Connecting to camera")
    camPort = open_camPort()

    #connect to drone
    debPrint("Connecting to drone")
    vehicle = connectToDrone()
    
    # runs the hangle_mount_status function after getting mavlink data (I hope it continualy updates gimble values) should run in asynchronous
    vehicle.add_message_listener('MOUNT_STATUS', handle_mount_status) 

    i = 0
    while not (the_mission_is_done_and_all_txt_files_are_saved):
        # drone gps position
        droneCoord = vehicle.location.global_relative_frame
        drone_lat = droneCoord.lat
        drone_lon = droneCoord.lon
        drone_alt = droneCoord.alt

        # drone angles
        attitude = vehicle.attitude
        drone_yaw = attitude.yaw # rad
        drone_pitch = attitude.pitch
        drone_roll = attitude.roll

        # Gimbal update
        vehicle.message_factory.mount_status_encode(0, 0, 0, 0, 0)
        
        # Process camera frame
        ret, frame = camPort.read()
        if ret:
            # Save temporary image for YOLO processing
            temp_img_path = "fire_detection.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            # Run YOLO detection
            base_name_path = save_yolo_detections(model, temp_img_path, yolo_txt_output_dir)

            telemetry_path = os.path.join(telemetry_output_dir, base_name_path)
            with open(telemetry_path, 'w') as f:
                f.write(f"{drone_lat}\n")
                f.write(f"{drone_lon}\n")
                f.write(f"{drone_alt}\n")
                f.write(f"{gimbal_pitch}\n")
                f.write(f"{gimbal_roll}\n")
                f.write(f"{gimbal_yaw}\n")
                f.write(f"{drone_pitch}\n")
                f.write(f"{drone_roll}\n")
                f.write(f"{drone_yaw}\n")

    vehicle.remove_message_listener('MOUNT_STATUS', handle_mount_status)
    camPort.release()
    vehicle.close()

    while (the_mission_is_done_and_all_txt_files_are_saved):
        # Localisation des feux et créeation du fichier KML
        localisationFeux(yolo_txt_output_dir, telemetry_output_dir)
        exit # or change mission plan

    #close everything
    debPrint("Program finished")
    camPort.close()
    vehicle.close()

def save_yolo_detections(model, image_path, yolo_txt_output_dir, index, class_mapping={"feux": 0}):
    """Save YOLO detections in format readable by read_yolo_annotations()"""
    os.makedirs(yolo_txt_output_dir, exist_ok=True)
    
    # Run inference
    results = model(image_path)
    
    # Make annotation file that will be passed to the fire position calculation (localisationFeux())
    base_name_path = os.path.splitext(os.path.basename(image_path))[0] + str(index)
    base_name_path = base_name_path + ".txt"
    txt_path = os.path.join(yolo_txt_output_dir, base_name_path)
    
    # Write bounding boxes to file
    with open(txt_path, 'w') as f:
        for box in results[0].boxes:
            class_id = int(box.cls)
            xywh_norm = box.xywhn[0].tolist()  # Normalized [x_center, y_center, width, height]
            
            # Write line in YOLO format
            line = f"{class_id} {xywh_norm[0]} {xywh_norm[1]} {xywh_norm[2]} {xywh_norm[3]}\n"
            f.write(line)
    
    return base_name_path

# Add MAVLink listener for gimbal
def handle_mount_status(msg):
    gimbal_pitch = msg.pointing_a / 100.0  # Convert centidegrees to degrees
    gimbal_roll = msg.pointing_b / 100.0
    gimbal_yaw = msg.pointing_c / 100.0

def open_camPort():
    #unused with usb camera
    global pargs
    if pargs.serialcamera:
        camConnected = False
        port = None
        while not camConnected:
            try:
                port = serial.Serial(pargs.serialcamera, baudrate=57600, timeout=1.0)
                camConnected = True
            except:
                time.sleep(5)
        return port
    else:
        return None
    
def connectToDrone():
    global pargs
    connected = False
    vehicle = None
    while not connected:
        try:
            vehicle = connect(pargs.dronelink, wait_ready=True)
            connected = True
        except:
            #wait before trying to connect again
            time.sleep(10)
    return vehicle


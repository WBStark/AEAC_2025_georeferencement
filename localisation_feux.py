import numpy as np
import os
from ultralytics import yolo

# KML generation function
def create_kml(positions, output_kml_path):
    kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
'''

    kml_footer = '''
</Document>
</kml>
'''

    # Start building the KML content
    kml_content = kml_header

    for idx, pos in enumerate(positions, 1):
        lat, lon, alt = pos
        placemark = f'''
<Placemark>
<name>Hotspot {idx}</name>
<Point>
<coordinates>{lon},{lat},{alt}</coordinates>
</Point>
</Placemark>
'''
        kml_content += placemark

    # Closing the KML document
    kml_content += kml_footer

    # Write to output KML file
    with open(output_kml_path, 'w') as kml_file:
        kml_file.write(kml_content)

# GPS functions

# Define constants with higher precision
a = 6378137.0  # semi-major axis in meters (WGS84)
b = 6356752.314245  # semi-minor axis in meters (WGS84)
e_sq = 6.694379990141316e-3  # WGS-84 first eccentricity squared
e1_2 = (a**2 - b**2) / b**2  # Second eccentricity squared (same as 1/e_sq)

# Function to convert GPS to ECEF (Earth-Centered, Earth-Fixed)
def geodetic_to_ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e_sq) * N + alt) * np.sin(lat)
    return np.array([x, y, z])

# Function to convert ECEF to GPS using Bowring's method with improved precision
def ecef_to_geodetic_bowring(x, y, z, max_iterations=10, tolerance=1e-15):
    def radius_of_curvature(latitude):
        return a / np.sqrt(1 - e_sq * np.sin(latitude)**2)

    # polar distance p and initial theta angle
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    latitude = np.arctan2(z + e1_2 * b * np.sin(theta)**3, p - e_sq * a * np.cos(theta)**3)
    
    # Bowring
    for _ in range(max_iterations):
        latitude_new = np.arctan2(z + e1_2 * b * np.sin(latitude)**3, p - e_sq * a * np.cos(latitude)**3)
        if np.abs(latitude_new - latitude) < tolerance:
            break
        latitude = latitude_new
    
    longitude = np.arctan2(y, x)
    
    N = radius_of_curvature(latitude)
    
    altitude = p / np.cos(latitude) - N

    return np.degrees(longitude), np.degrees(latitude), altitude


def fire_img_position_normalized(
    box, focal_length, sensor_width, sensor_height
):

    x_center_norm = box["x_center"]
    y_center_norm = box["y_center"]

    # Calculates ray from camera to position of fire in image frame (3D ray onto 2D plane)
    x_offset = (x_center_norm - 1 / 2) * sensor_width 
    y_offset = (y_center_norm - 1 / 2) * sensor_height 

    ray_cam = [x_offset, y_offset, focal_length]
    ray_cam /= np.linalg.norm(ray_cam)  # Normalize

    return ray_cam

# Function to calculate GPS position of a point in the image
def vehical_rot_matrix(drone_roll, drone_pitch, drone_yaw, gimbal_roll, gimbal_pitch, gimbal_yaw):
    roll = drone_roll + gimbal_roll
    pitch = drone_pitch + gimbal_pitch
    yaw = drone_yaw + gimbal_yaw
    # Yaw (about Z-axis (up))
    R_azimuth = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Pitch (about X-axis (east))
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float64)

    # Roll (about Y-axis (north))
    R_roll = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ], dtype=np.float64)
    
    # ENU rotation matrix
    vehical_rot_matrix = np.dot(R_azimuth, np.dot(R_pitch, R_roll))
    
    return vehical_rot_matrix

def read_yolo_annotations(yolo_file, class_mapping):
    bounding_boxes = []
    
    with open(yolo_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])  # First element is the class ID
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Create a dictionary for the bounding box with specified keys
            bounding_box_entry = {
                "class": class_id,
                "label": 0,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            }
            
            bounding_boxes.append(bounding_box_entry)
    
    return bounding_boxes

def localisationFeux(yolo_annotation_folder, drone_telemetry_folder):
    #We will need to read the files that hold the gps positions of the drone, gimbal angles and drone angles, when a fire is dectected
    #to calculate where that fire is positionned in gps_decimal values

    # ALSO WE NEED TO KNOW IF THE DRONE STARTS AS ENU POSITION (EAST, NORTH, UP VECTOR) im not sure lol
    positions = [] # To store the GPS positions for the KML think yourcelf uwu 

    focal_length = 76 # mm
    sensor_width = 7.41 # mm 
    sensor_height = 5.56  # mm 

    for yolo_file_name in sorted([f for f in os.listdir(yolo_annotation_folder) if f.endswith(('.txt'))]):
        yolo_path = os.path.join(yolo_annotation_folder, yolo_file_name)
        telemetry_path = os.path.join(drone_telemetry_folder, yolo_file_name)
        
        # Read YOLO annotations
        bounding_boxes = read_yolo_annotations(yolo_path)
        
        # Read telemetry data
        try:
            with open(telemetry_path, 'r') as f:
                telemetry_lines = f.readlines()
                if len(telemetry_lines) >= 9:  # Ensure we have all expected values
                    drone_lat = float(telemetry_lines[0].strip())
                    drone_lon = float(telemetry_lines[1].strip())
                    drone_alt = float(telemetry_lines[2].strip())
                    gimbal_pitch = float(telemetry_lines[3].strip())
                    gimbal_roll = float(telemetry_lines[4].strip())
                    gimbal_yaw = float(telemetry_lines[5].strip())
                    drone_pitch = float(telemetry_lines[6].strip())
                    drone_roll = float(telemetry_lines[7].strip())
                    drone_yaw = float(telemetry_lines[8].strip())
        except (FileNotFoundError, IndexError, ValueError) as e:
            print(f"Error reading telemetry for {yolo_file_name}: {e}")
            continue

        # Obtaining ground distance in ecef
        # or you could just do gps_alt_drone - gps_alt_ground and switch it to ecef after for t calculation


        #/////////////////////////////////////////////#
        ground_ecef = geodetic_to_ecef(drone_lat, drone_lon, GROUND) #///// put the ground gps position of the drone at ground level///////
        #/////////////////////////////////////////////#


        ground_alt = ground_ecef[2]


        for box in bounding_boxes:
            # Find the vector pointing from the camera to the dectect dot in the image plane (camera frame)
            normalized_ray_cam = fire_img_position_normalized(box, focal_length, sensor_width, sensor_height)

            # Rotation matrix to put the detected object in the world frame
            vehical_rot_matrix = vehical_rot_matrix(drone_roll, drone_pitch, drone_yaw, gimbal_roll, gimbal_pitch, gimbal_yaw)
            ray2fire_world_frame = np.dot(vehical_rot_matrix, normalized_ray_cam)

            drone_ecef = geodetic_to_ecef(drone_lat, drone_lon, drone_alt)

            # We want to scale the ray2fire_world_frame vector to real size in ecef, 
            # so we find the scaling facter t using the height, finding the difference between real height difference and normalized height in the vector
            t = (drone_ecef[2] - ground_alt) / ray2fire_world_frame[2]
            # Here we find the x, y, z positions on the ground where the ray touches the ground. so we know the ground position of the fire
            fire_in_world_ecef = drone_ecef + t*ray2fire_world_frame

            # GPS coordinates of the detected fire in the image
            fire_in_world_geodetic = ecef_to_geodetic_bowring(fire_in_world_ecef[0], fire_in_world_ecef[1], fire_in_world_ecef[2])

            positions.append(fire_in_world_geodetic)  # Append the GPS coordinates

    # After processing all images and bounding boxes, create the KML file
    output_kml = "output.kml"
    create_kml(positions, output_kml)


    """ Not needed for now
# Function to convert ECEF to ENU (East, North, Up)
def ecef_to_enu(drone_lat, drone_lon, drone_alt, point_ecef):
    lat_rad = np.radians(drone_lat)
    lon_rad = np.radians(drone_lon)

    # Compute the ECEF position of the drone
    drone_ecef = geodetic_to_ecef(drone_lat, drone_lon, drone_alt)

    # Calculate the vector from the drone to the point in ECEF
    ecef_vector = point_ecef - drone_ecef

    # Rotation matrix from ECEF to ENU
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ], dtype=np.float64)

    # Rotate the ECEF vector to the ENU frame
    enu_vector = np.dot(R, ecef_vector)

    return enu_vector

# Function to convert ENU to ECEF with matrix multiplication (maintaining precision)
def enu_to_ecef(drone_lat, drone_lon, drone_alt, enu_vector):
    drone_ecef = geodetic_to_ecef(drone_lat, drone_lon, drone_alt)

    # Convert drone geodetic coordinates to radians
    lat_rad = np.radians(drone_lat)
    lon_rad = np.radians(drone_lon)

    # Rotation matrix from ENU to ECEF (transpose of ECEF-to-ENU)
    R = np.array([
        [-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
        [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
        [0, np.cos(lat_rad), np.sin(lat_rad)]
    ], dtype=np.float64)

    # Rotate the ENU vector back to ECEF
    ecef_offset = np.dot(R, enu_vector)

    # Add the offset to the ECEF position of the drone
    ecef_vector = drone_ecef + ecef_offset

    return ecef_vector
"""

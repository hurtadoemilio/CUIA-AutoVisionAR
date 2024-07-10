import sys
import cv2
import numpy as np
import os
import csv
import speech_recognition as sr
import logging
import pygame
import requests
from flask import current_app
from config import CAR_DATA_FILE, CAR_IMAGES_FOLDER, ARUCO_DICT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize pygame mixer
pygame.mixer.init()
fast_car_sound = pygame.mixer.Sound('data/fastcar.wav')


def load_car_data():
    """
    Load car data from a CSV file.

    Returns:
        dict: A dictionary with marker IDs as keys and car details as values.
    """
    car_data = {}
    try:
        with open(CAR_DATA_FILE, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                marker_id = int(row['marker_id'])
                car_data[marker_id] = {
                    'image_filename': row['image_filename'],
                    'model': row['model'],
                    'price': row['price'],
                    'engine': row['engine']
                }
        logger.info("Car data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading car data: {str(e)}")
    return car_data


def load_car_images(car_data):
    """
    Load car images based on the car data.

    Args:
        car_data (dict): A dictionary containing car data.

    Returns:
        dict: A dictionary with marker IDs as keys and image paths as values.
    """
    car_images = {}
    try:
        for marker_id, data in car_data.items():
            image_path = os.path.join(CAR_IMAGES_FOLDER, data['image_filename'])
            car_images[marker_id] = image_path
        logger.info("Car images loaded successfully")
    except Exception as e:
        logger.error(f"Error loading car images: {str(e)}")
    return car_images


def detect_markers(frame):
    """
    Detect ArUco markers in the given frame.

    Args:
        frame (numpy.ndarray): The image frame in which to detect markers.

    Returns:
        tuple: The corners and IDs of detected markers.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector.detectMarkers(frame)


def overlay_image_on_marker(frame, corners, image_path):
    """
    Overlay an image on the detected marker in the frame, centering it on the marker.

    Args:
        frame (numpy.ndarray): The image frame.
        corners (numpy.ndarray): The corners of the detected marker.
        image_path (str): The path to the image to overlay.

    Returns:
        numpy.ndarray: The frame with the overlay image.
    """
    overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        logger.error(f"Error: Could not load image at path {image_path}")
        return frame

    # Calculate the center of the marker
    marker_center = np.mean(corners[0], axis=0).astype(int)

    # Get dimensions of the overlay image
    oh, ow = overlay_img.shape[:2]

    # Calculate top-left corner of where to place the image
    top_left_x = int(marker_center[0] - ow / 2)
    top_left_y = int(marker_center[1] - oh / 2)

    # Ensure the overlay fits within the frame
    frame_h, frame_w = frame.shape[:2]
    bottom_right_x = min(top_left_x + ow, frame_w)
    bottom_right_y = min(top_left_y + oh, frame_h)
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)

    # Calculate the actual width and height of the overlay
    overlay_w = bottom_right_x - top_left_x
    overlay_h = bottom_right_y - top_left_y

    # Resize the overlay image to fit within the frame
    overlay_resized = cv2.resize(overlay_img, (overlay_w, overlay_h))

    # Create a region of interest (ROI) in the frame
    roi = frame[top_left_y:top_left_y + overlay_h, top_left_x:top_left_x + overlay_w]

    # For images with an alpha channel
    if overlay_resized.shape[2] == 4:
        overlay_colors = overlay_resized[:, :, :3]
        overlay_alpha = overlay_resized[:, :, 3] / 255.0
        alpha_3d = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))

        # Blend the overlay with the ROI
        blended_roi = (1 - alpha_3d) * roi + alpha_3d * overlay_colors
        frame[top_left_y:top_left_y + overlay_h, top_left_x:top_left_x + overlay_w] = blended_roi
    else:
        # If no alpha channel, simply copy the overlay onto the frame
        frame[top_left_y:top_left_y + overlay_h, top_left_x:top_left_x + overlay_w] = overlay_resized

    return frame


def draw_3d_text_below_marker(frame, text, corners, font, font_scale, color, thickness, depth):
    """
    Draw 3D text below the detected marker in the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        text (str): The text to draw.
        corners (numpy.ndarray): The corners of the detected marker.
        font (int): The font type.
        font_scale (float): The scale of the font.
        color (tuple): The color of the text.
        thickness (int): The thickness of the text.
        depth (int): The depth of the 3D effect.
    """
    bottom_left = corners[0][3]
    bottom_right = corners[0][2]
    bottom_center = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int(bottom_center[0] - text_width / 2)
    text_y = int(bottom_center[1] + text_height + 10)

    # Draw shadow for 3D effect
    for i in range(depth):
        offset = i * 2
        cv2.putText(frame, text, (text_x + offset, text_y + offset), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def generate_frames(camera, car_data, car_images, show_model_for_marker, show_price_for_marker, show_engine_for_marker,
                    discovered_cars):
    """
    Generate frames from the camera feed, overlay images on detected markers, and draw text.

    Args:
        camera (cv2.VideoCapture): The camera object.
        car_data (dict): A dictionary containing car data.
        car_images (dict): A dictionary containing car image paths.
        show_model_for_marker (dict): A dictionary indicating whether to show the model for each marker.
        show_price_for_marker (dict): A dictionary indicating whether to show the price for each marker.
        show_engine_for_marker (dict): A dictionary indicating whether to show the engine info for each marker.
        discovered_cars (set): A set of discovered car marker IDs.

    Yields:
        bytes: The frame data in JPEG format.
    """
    fast_cars_discovered = set()  # Keep track of fast cars we've already played the sound for

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            corners, ids, _ = detect_markers(frame)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids):
                    marker_id = marker_id[0]
                    if marker_id in car_images:
                        frame = overlay_image_on_marker(frame, corners[i], car_images[marker_id])
                    if marker_id in car_data:
                        if marker_id not in discovered_cars:
                            discovered_cars.add(marker_id)
                            # Check if it's a fast car (more than 500 CV)
                            engine = car_data[marker_id]['engine']
                            try:
                                engine_power = int(engine.split()[0])  # Assuming engine format is "500 CV"
                                if engine_power > 500 and marker_id not in fast_cars_discovered:
                                    fast_car_sound.play()
                                    fast_cars_discovered.add(marker_id)
                            except ValueError:
                                logger.error(f"Error parsing engine power for marker {marker_id}")

                        display_text = f"ID: {marker_id}"
                        if show_model_for_marker.get(marker_id, False):
                            model = car_data[marker_id]['model']
                            display_text += f", Modelo: {model}"
                        if show_price_for_marker.get(marker_id, False):
                            price = car_data[marker_id]['price']
                            display_text += f", Precio: {price}"
                        if show_engine_for_marker.get(marker_id, False):
                            engine = car_data[marker_id]['engine']
                            display_text += f", Motor: {engine}"
                        draw_3d_text_below_marker(frame, display_text, corners[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                  (0, 255, 0), 2, 5)
                    else:
                        draw_3d_text_below_marker(frame, f"Marker Detected! ID: {marker_id}", corners[i],
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 5)
            else:
                cv2.putText(frame, "No marker detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


def listen_for_commands(car_data, show_model_for_marker, show_price_for_marker, show_engine_for_marker, stop_event,
                        shutdown_event, server_url):
    """
    Listen for voice commands and perform actions based on the commands.

    Args:
        car_data (dict): A dictionary containing car data.
        show_model_for_marker (dict): A dictionary indicating whether to show the model for each marker.
        show_price_for_marker (dict): A dictionary indicating whether to show the price for each marker.
        show_engine_for_marker (dict): A dictionary indicating whether to show the engine info for each marker.
        stop_event (threading.Event): An event to stop the command listener.
        shutdown_event (threading.Event): An event to shut down the server.
        server_url (str): The URL of the server.
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while not stop_event.is_set():
        with microphone as source:
            logger.info("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio, language="es-ES").lower()
            logger.info(f"Command received: {command}")

            if "salir" in command:
                logger.info("Salir command received. Shutting down the server.")
                shutdown_event.set()
                stop_event.set()
                return

            elif "modelo" in command:
                if any(str(id) in command for id in car_data.keys()):
                    for id in car_data.keys():
                        if str(id) in command:
                            result = toggle_model(id, show_model_for_marker, car_data)
                            logger.info(result)
                            break
                else:
                    result = toggle_model(None, show_model_for_marker, car_data)
                    logger.info(result)

            elif "precio" in command:
                if any(str(id) in command for id in car_data.keys()):
                    for id in car_data.keys():
                        if str(id) in command:
                            result = show_price(id, show_price_for_marker)
                            logger.info(result)
                            break
                else:
                    result = show_all_prices(show_price_for_marker, car_data)
                    logger.info(result)

            elif "motor" in command:
                if any(str(id) in command for id in car_data.keys()):
                    for id in car_data.keys():
                        if str(id) in command:
                            result = toggle_engine_info(id, show_engine_for_marker, car_data)
                            logger.info(result)
                            break
                else:
                    result = toggle_engine_info(None, show_engine_for_marker, car_data)
                    logger.info(result)

        except sr.UnknownValueError:
            logger.error("Could not understand the audio")
        except sr.RequestError as e:
            logger.error(f"Could not request results; {e}")

    logger.info("Voice command listener has stopped")


def toggle_model(marker_id, show_model_for_marker, car_data):
    """
    Toggle the display of the car model for a specific marker or all markers.

    Args:
        marker_id (int or None): The marker ID to toggle the model display for. If None, toggles for all markers.
        show_model_for_marker (dict): A dictionary indicating whether to show the model for each marker.
        car_data (dict): A dictionary containing car data.

    Returns:
        str: A message indicating the result of the toggle operation.
    """
    if marker_id is None:
        for id in car_data.keys():
            show_model_for_marker[id] = not show_model_for_marker.get(id, False)
        return "Model display toggled for all markers"
    else:
        show_model_for_marker[marker_id] = not show_model_for_marker.get(marker_id, False)
        return f"Model display for marker {marker_id} is now {'on' if show_model_for_marker[marker_id] else 'off'}"


def toggle_engine_info(marker_id, show_engine_for_marker, car_data):
    """
    Toggle the display of the car engine information for a specific marker or all markers.

    Args:
        marker_id (int or None): The marker ID to toggle the engine info display for. If None, toggles for all markers.
        show_engine_for_marker (dict): A dictionary indicating whether to show the engine info for each marker.
        car_data (dict): A dictionary containing car data.

    Returns:
        str: A message indicating the result of the toggle operation.
    """
    if marker_id is None:
        for id in car_data.keys():
            show_engine_for_marker[id] = not show_engine_for_marker.get(id, False)
        return "Engine information display toggled for all markers"
    else:
        show_engine_for_marker[marker_id] = not show_engine_for_marker.get(marker_id, False)
        return f"Engine information display for marker {marker_id} is now {'on' if show_engine_for_marker[marker_id] else 'off'}"


def show_price(marker_id, show_price_for_marker):
    """
    Toggle the display of the car price for a specific marker.

    Args:
        marker_id (int): The marker ID to toggle the price display for.
        show_price_for_marker (dict): A dictionary indicating whether to show the price for each marker.

    Returns:
        str: A message indicating the result of the toggle operation.
    """
    show_price_for_marker[marker_id] = not show_price_for_marker.get(marker_id, False)
    return f"Price display for marker {marker_id} is now {'on' if show_price_for_marker[marker_id] else 'off'}"


def show_all_prices(show_price_for_marker, car_data):
    """
    Toggle the display of car prices for all markers.

    Args:
        show_price_for_marker (dict): A dictionary indicating whether to show the price for each marker.
        car_data (dict): A dictionary containing car data.

    Returns:
        str: A message indicating the result of the toggle operation.
    """
    for id in car_data.keys():
        show_price_for_marker[id] = not show_price_for_marker.get(id, False)
    return "Price display toggled for all markers"

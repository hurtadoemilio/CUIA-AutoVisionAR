# app.py
import os
import threading
import time

import cv2
import numpy as np
import requests
from flask import (
    Flask,
    Response,
    copy_current_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from functools import wraps

from config import CAMERA_INDEX
from face_recognition_utils import FaceRecognition
from user_discoveries import add_discovery, get_all_discoveries, get_user_discoveries
from utils import (
    generate_frames,
    listen_for_commands,
    load_car_data,
    load_car_images,
)

# Initialize the Flask application
app = Flask(__name__)

app.secret_key = os.urandom(24)

# Initialize the camera with the specified index
camera = cv2.VideoCapture(CAMERA_INDEX)

# Load car data and images
car_data = load_car_data()
car_images = load_car_images(car_data)

# Dictionaries to keep track of which markers should display model, price and engine data
show_model_for_marker = {}
show_price_for_marker = {}
show_engine_for_marker = {}

# Set to keep track of discovered cars
discovered_cars = set()

# Flag to ensure the voice command thread is started only once
voice_thread_started = False
voice_command_stop_event = threading.Event()
shutdown_event = threading.Event()

# Face login
face_recognizer = FaceRecognition()


def load_users():
    loaded_users = {}
    try:
        with open('data/users.txt', 'r') as file:
            for line in file:
                username, encoded_password = line.strip().split(':')
                loaded_users[username] = encoded_password
    except FileNotFoundError:
        pass
    return loaded_users


users = load_users()


def encode_password(password):
    return ''.join([chr((ord(c) - 97 + 13) % 26 + 97) for c in password.lower()])


def decode_password(encoded_password):
    return ''.join([chr((ord(c) - 97 - 13) % 26 + 97) for c in encoded_password.lower()])


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and encode_password(password) == users[username]:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('register.html', error='Username already exists')
        else:
            users[username] = encode_password(password)
            with open('data/users.txt', 'a') as file:
                file.write(f"{username}:{encode_password(password)}\n")
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/')
@login_required
def home():
    """
    Render the home page.
    """
    username = session.get('username', 'Invitado')
    return render_template('index.html', username=username)


@app.route('/manual')
@login_required
def manual():
    """
    Render the manual page with steps for using the application.
    """
    steps = [
        "Abre la aplicación y pulsa el botón Cámara.",
        "Apunta con la cámara a un marcador de aRuco.",
        "La aplicación detectará el marcador y mostrará el coche",
        "Usa el botón 'Home' para volver a la pantalla principal en cualquier momento."
    ]
    return render_template('manual.html', steps=steps)


@app.route('/camara')
@login_required
def camara():
    global voice_thread_started
    if not voice_thread_started:
        server_url = request.url_root.rstrip('/')
        # Start a new thread for listening to voice commands
        threading.Thread(
            target=listen_for_commands,
            args=(
                car_data, show_model_for_marker, show_price_for_marker, show_engine_for_marker,
                voice_command_stop_event,
                shutdown_event, server_url),
            daemon=True
        ).start()
        voice_thread_started = True
    return render_template('camara.html')


@app.route('/video_feed')
@login_required
def video_feed():
    current_user = session.get('username')
    user_discoveries = set(get_user_discoveries(current_user))

    @copy_current_request_context
    def generate():
        for frame in generate_frames(camera, car_data, car_images, show_model_for_marker, show_price_for_marker,
                                     show_engine_for_marker, discovered_cars):
            new_discoveries = discovered_cars - user_discoveries
            if new_discoveries:
                for car_id in new_discoveries:
                    try:
                        add_discovery(current_user, car_id)
                    except Exception as e:
                        print(f"Error adding discovery: {str(e)}")
                user_discoveries.update(new_discoveries)
            yield frame

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/discovered_cars')
@login_required
def discovered_cars_page():
    """
    Render the page showing details of discovered cars for the current user.
    """
    current_user = session.get('username')
    user_discoveries = get_user_discoveries(current_user)
    discovered_car_data = {id: car_data[id] for id in user_discoveries if id in car_data}
    return render_template('discovered_cars.html', discovered_cars=discovered_car_data)


@app.route('/all_discoveries')
@login_required
def all_discoveries_page():
    """
    Render a page showing all discoveries by all users (admin view).
    """
    all_discoveries = get_all_discoveries()
    return render_template('all_discoveries.html', all_discoveries=all_discoveries, car_data=car_data)


@app.route('/stop_voice_commands')
@login_required
def stop_voice_commands():
    voice_command_stop_event.set()
    return "Voice command listener is stopping"


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_event.set()
    return 'Server is shutting down...'


def run_app():
    app.run(host='localhost', port=5000, debug=False, use_reloader=False, threaded=True)


@app.route('/facial_login', methods=['GET', 'POST'])
def facial_login():
    return render_template('facial_login.html')


@app.route('/facial_recognition_feed')
def facial_recognition_feed():
    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Process the frame with facial recognition
                frame = face_recognizer.process_frame(frame)

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process_facial_login', methods=['POST'])
def process_facial_login():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file:
        # Read the image file
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the frame with facial recognition
        processed_frame = face_recognizer.process_frame(frame)

        # Check if any known face was detected
        if any(name != 'Unknown' for name in face_recognizer.face_names):
            # Get the first known face (you might want to modify this logic)
            username = next(name.split()[0] for name in face_recognizer.face_names if name != 'Unknown')

            if username in users:
                session['username'] = username
                return jsonify({'success': True})

        return jsonify({'success': False, 'error': 'Face not recognized or user not registered'})

    return jsonify({'success': False, 'error': 'Invalid file'})


if __name__ == '__main__':
    server_thread = threading.Thread(target=run_app)
    server_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    finally:
        print("Shutting down the application...")
        shutdown_event.set()
        voice_command_stop_event.set()  # Stop the voice command thread

        # Give some time for the voice command thread to stop
        time.sleep(2)

        # Attempt to shut down the Flask server
        try:
            requests.post('http://localhost:5000/shutdown', timeout=5)
        except requests.exceptions.RequestException:
            print("Could not send shutdown request to server")

        # Wait for the server thread to finish
        server_thread.join(timeout=5)

        # Close the camera
        camera.release()

        print("Application shutdown complete.")
        os._exit()

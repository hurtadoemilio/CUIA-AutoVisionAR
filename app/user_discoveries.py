import json
import os
import numpy as np

DISCOVERIES_FILE = 'data/discoveries.json'


def load_user_discoveries():
    """
    Load user discoveries from the JSON file.

    Returns:
        dict: A dictionary containing user discoveries. If the file is empty or contains invalid JSON,
              an empty dictionary is returned.
    """
    try:
        with open(DISCOVERIES_FILE, 'r') as file:
            content = file.read()
            if not content:
                return {}  # Return an empty dict if the file is empty
            discoveries = json.loads(content)
            # Ensure discoveries is a dictionary
            return discoveries if isinstance(discoveries, dict) else {}
    except json.JSONDecodeError:
        print("Error: The discoveries.json file contains invalid JSON. Resetting to an empty dictionary.")
        return {}
    except FileNotFoundError:
        print("Info: The discoveries.json file doesn't exist. Creating a new one.")
        return {}


def save_user_discoveries(discoveries):
    """
    Save user discoveries to the JSON file.

    Args:
        discoveries (dict): A dictionary containing user discoveries to be saved.

    Note:
        If the input is not a dictionary, it will be reset to an empty dictionary before saving.
    """
    if not isinstance(discoveries, dict):
        print("Error: Attempting to save non-dictionary data. Resetting to an empty dictionary.")
        discoveries = {}
    with open(DISCOVERIES_FILE, 'w') as file:
        json.dump(discoveries, file, indent=2)


def add_discovery(username, car_id):
    """
    Add a car discovery for a specific user.

    Args:
        username (str): The username of the user making the discovery.
        car_id (int or str): The ID of the car discovered.

    Note:
        The car_id is converted to an integer before being added to the user's discoveries.
    """
    discoveries = load_user_discoveries()
    if username not in discoveries:
        discoveries[username] = []
    if car_id not in discoveries[username]:
        discoveries[username].append(int(car_id))  # Ensure car_id is a native Python int
    save_user_discoveries(discoveries)


def get_user_discoveries(username):
    """
    Get the list of car IDs discovered by a specific user.

    Args:
        username (str): The username of the user whose discoveries are to be retrieved.

    Returns:
        list: A list of car IDs discovered by the user. Returns an empty list if the user has no discoveries.
    """
    discoveries = load_user_discoveries()
    return discoveries.get(username, [])


def get_all_discoveries():
    """
    Get all user discoveries.

    Returns:
        dict: A dictionary containing all user discoveries, where keys are usernames and values are lists of car IDs.
    """
    return load_user_discoveries()

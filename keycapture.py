from pynput import keyboard
import multiprocessing
import threading

# Initialize a manager to create a shared variable
manager = multiprocessing.Manager()
shared_keys = manager.list()
shared_keys_lock = threading.Lock()

# Function to handle key press events
def on_key_press(key):
    try:
        char = key.char
        if char not in shared_keys:
            with shared_keys_lock:
        # Add the currently pressed key to the shared list
                shared_keys.append(key.char)
    except AttributeError:
        # Handle special keys here if needed
        pass

# Function to handle key release events
def on_key_release(key):
    try:
        char=key.char
        if char:
            with shared_keys_lock:# Remove the released key from the shared list
                shared_keys.remove(key.char)
    except (KeyError, AttributeError):
        pass

# Create keyboard listener
def listen_for_keys():
    with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        listener.join()

# Function to get the currently pressed keys
def get_current_keys():
    return set(shared_keys)

from pynput import keyboard
import threading

# Initialize a set to store the currently pressed keys
current_keys = set()

# Function to handle key press events
def on_key_press(key):
    global current_keys
    try:
        # Add the currently pressed key to the set
        current_keys.add(key.char)
    except AttributeError:
        # Handle special keys here if needed
        pass

# Function to handle key release events
def on_key_release(key):
    global current_keys
    # Remove the released key from the set
    try:
        current_keys.remove(key.char)
    except KeyError:
        pass

# Create keyboard listener in a separate thread
def listen_for_keys():
    with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        listener.join()

# Start the key listener thread
key_listener_thread = threading.Thread(target=listen_for_keys)
key_listener_thread.daemon = True
key_listener_thread.start()

# In your main loop, you can join the currently pressed keys into a single string.
while True:
    if current_keys:
        # Convert the set of keys to a string and print it
        current_keys_string = ''.join(current_keys)
        print(f"Currently pressed keys: {current_keys_string}")
        # You can use current_keys_string as needed in your loop
    else:
        # No keys are currently pressed
        pass

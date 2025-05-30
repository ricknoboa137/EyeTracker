import time
import threading
import csv
import os
from datetime import datetime
from collections import deque

from pynput import keyboard, mouse

class InputLogger:
    """
    Captures global keyboard and mouse activity, including mouse position,
    and logs it to a CSV file. Events are timestamped.
    """
    def __init__(self, log_file_path="input_log.csv"):
        self.log_file_path = log_file_path
        self.event_queue = deque() # Thread-safe queue for events
        self.stop_event = threading.Event() # Event to signal logging thread to stop
        self.logging_thread = None
        self.csv_file = None
        self.csv_writer = None
        self.mouse_controller = mouse.Controller() # To get current mouse position

        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initializes the CSV log file with a header if it doesn't exist."""
        try:
            # 'a' mode to append, 'newline=''' to prevent extra blank rows
            self.csv_file = open(self.log_file_path, 'a', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header only if file is empty (or new)
            if os.stat(self.log_file_path).st_size == 0:
                self.csv_writer.writerow([
                    "timestamp", "event_type", "key", "button", "scroll_dx", "scroll_dy",
                    "mouse_x", "mouse_y", "event_state"
                ])
            print(f"Logging to {self.log_file_path}")
        except IOError as e:
            print(f"Error opening log file '{self.log_file_path}': {e}")
            self.csv_file = None
            self.csv_writer = None

    def _get_current_mouse_position(self):
        """Returns the current mouse (x, y) coordinates."""
        return self.mouse_controller.position

    def _format_event(self, event_type, **kwargs):
        """
        Formats event data into a consistent dictionary for CSV logging.
        Includes timestamp and current mouse position for all events.
        """
        current_time = time.time()
        mouse_x, mouse_y = self._get_current_mouse_position()

        data = {
            "timestamp": current_time,
            "event_type": event_type,
            "key": "",          # For keyboard events
            "button": "",       # For mouse click events
            "scroll_dx": "",    # For mouse scroll events
            "scroll_dy": "",    # For mouse scroll events
            "mouse_x": mouse_x, # Mouse X coordinate at the time of the event
            "mouse_y": mouse_y, # Mouse Y coordinate at the time of the event
            "event_state": ""   # e.g., "pressed", "released", "moved", "clicked", "scrolled"
        }
        data.update(kwargs) # Override defaults with specific event data
        return data

    def _on_press(self, key):
        """Callback for keyboard key press events."""
        try:
            key_char = str(key.char) # Regular characters
        except AttributeError:
            key_char = str(key)      # Special keys (e.g., Key.space, Key.esc)
        
        event_data = self._format_event(
            event_type="keyboard",
            key=key_char,
            event_state="pressed"
        )
        self.event_queue.append(event_data)

    def _on_release(self, key):
        """Callback for keyboard key release events."""
        try:
            key_char = str(key.char)
        except AttributeError:
            key_char = str(key)
        
        event_data = self._format_event(
            event_type="keyboard",
            key=key_char,
            event_state="released"
        )
        self.event_queue.append(event_data)
        
        # Optionally, stop the listener if 'Esc' key is released
        if key == keyboard.Key.esc:
            print("Escape key pressed. Stopping input logger...")
            return False # This stops the keyboard listener

    def _on_move(self, x, y):
        """Callback for mouse movement events."""
        event_data = self._format_event(
            event_type="mouse_move", # Differentiate from click/scroll
            mouse_x=x,
            mouse_y=y,
            event_state="moved"
        )
        self.event_queue.append(event_data)

    def _on_click(self, x, y, button, pressed):
        """Callback for mouse click events."""
        event_state = "pressed" if pressed else "released"
        event_data = self._format_event(
            event_type="mouse_click",
            button=str(button),
            mouse_x=x,
            mouse_y=y,
            event_state=event_state
        )
        self.event_queue.append(event_data)

    def _on_scroll(self, x, y, dx, dy):
        """Callback for mouse scroll events."""
        event_data = self._format_event(
            event_type="mouse_scroll",
            scroll_dx=dx,
            scroll_dy=dy,
            mouse_x=x,
            mouse_y=y,
            event_state="scrolled"
        )
        self.event_queue.append(event_data)

    def _logging_worker(self):
        """
        Worker thread to write events from the queue to the CSV file.
        Runs in a separate thread to avoid blocking input listeners.
        """
        while not self.stop_event.is_set() or self.event_queue:
            if self.event_queue:
                event = self.event_queue.popleft()
                if self.csv_writer:
                    try:
                        self.csv_writer.writerow([
                            event["timestamp"], event["event_type"], event["key"],
                            event["button"], event["scroll_dx"], event["scroll_dy"],
                            event["mouse_x"], event["mouse_y"], event["event_state"]
                        ])
                        self.csv_file.flush() # Ensure data is written to disk immediately
                    except Exception as e:
                        print(f"Error writing to CSV: {e}")
            else:
                time.sleep(0.01) # Small sleep to prevent busy-waiting

    def start(self):
        """Starts the keyboard and mouse listeners and the logging thread."""
        if self.csv_file is None:
            print("Error: Log file not initialized. Cannot start logging. Check file path or permissions.")
            return

        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll
        )

        self.logging_thread = threading.Thread(target=self._logging_worker, name="LoggingThread")
        self.logging_thread.daemon = True # Allow main program to exit even if this thread is running

        print("Starting input listeners...")
        self.keyboard_listener.start()
        self.mouse_listener.start()
        self.logging_thread.start()
        print("Input logging active. Press 'Esc' key to stop the keyboard listener and exit.")

    def stop(self):
        """Stops all listeners and the logging thread gracefully."""
        print("Initiating graceful shutdown of input logger...")
        
        self.stop_event.set() # Signal logging thread to stop processing new events
        
        # Stop pynput listeners
        if self.keyboard_listener.running:
            self.keyboard_listener.stop()
        if self.mouse_listener.running:
            self.mouse_listener.stop()

        # Wait for the logging thread to finish processing remaining events in the queue
        if self.logging_thread and self.logging_thread.is_alive():
            print("Waiting for logging thread to finish...")
            self.logging_thread.join(timeout=5) # Give it some time to finish
            if self.logging_thread.is_alive():
                print("Warning: Logging thread did not terminate gracefully within timeout.")

        # Close the CSV file
        if self.csv_file:
            self.csv_file.close()
            print("Log file closed.")
        print("Input logger stopped.")

# Example Usage
if __name__ == "__main__":
    # Define a log directory and file name based on current timestamp
    log_dir = "input_logs"
    log_file_name = f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_log_path = os.path.join(log_dir, log_file_name)

    logger = InputLogger(log_file_path=full_log_path)
    logger.start()

    try:
        # Keep the main thread alive. The keyboard listener's on_release for 'esc'
        # will stop the keyboard listener itself, which will then allow the 
        # 'logger.keyboard_listener.join()' call to complete, triggering the finally block.
        # Alternatively, you can add a simple time.sleep(large_number) or
        # run a dummy loop to keep the main thread active.
        
        # This line effectively waits for the keyboard listener to stop (e.g., by pressing Esc)
        logger.keyboard_listener.join() 
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nCtrl+C detected. Shutting down input logger.")
    finally:
        # Ensure stop is called even if an exception occurs or program exits
        logger.stop()
        print("Application finished.")
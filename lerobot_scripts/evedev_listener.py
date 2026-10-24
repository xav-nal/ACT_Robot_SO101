import threading
from evdev import InputDevice, categorize, ecodes


def init_evdev_keyboard_listener(device_path: str):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "stop":False
    }

    device = InputDevice(device_path)

    def listen():
        for event in device.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            key = categorize(event)
            if key.keystate != key.key_down:
                continue

            keycode = key.keycode
            if isinstance(keycode, list):
                keycode = keycode[0]

            if keycode == "KEY_RIGHT":
                print("Right arrow pressed. Exiting loop...")
                events["exit_early"] = True

            elif keycode == "KEY_LEFT":
                print("Left arrow pressed. Re-recording episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True

            elif keycode == "KEY_ESC":
                print("Escape pressed. Stopping...")
                events["stop_recording"] = True
                events["exit_early"] = True
                break

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return device, events
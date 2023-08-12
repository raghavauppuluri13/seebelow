from devices import ForceSensor, RealsenseCapture
from utils import Hz
import time

if __name__ == "__main__":

    # rs_cap = RealsenseCapture()
    fs = ForceSensor()

    hz = Hz(print_hz=True)

    while True:
        # pcd = rs_cap.read()
        Fxyz = fs.read()
        if Fxyz is not None:
            print(Fxyz)
            hz.clock()
        time.sleep(0.002)

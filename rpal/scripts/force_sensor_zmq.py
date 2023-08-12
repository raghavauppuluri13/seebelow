from devices import ForceSensor
import time
from utils import Hz

from force_sensor_pb2 import ForceSensorMessage

import yaml
import argparse

import zmq


def sensor_data_to_force_msg(sensor_data) -> ForceSensorMessage:
    msg = ForceSensorMessage()
    msg.fx = sensor_data[0]
    msg.fy = sensor_data[1]
    msg.fz = sensor_data[2]
    return msg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg_path",
        help="path to config file",
        type=str,
        default="../../config/an-an-force.yml",
    )

    args = vars(parser.parse_args())
    with open(args["cfg_path"], "r") as f:
        params = yaml.safe_load(f)

    port = params["PC"]["FORCE_SENSOR_PUB_PORT"]
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{port}")

    f = ForceSensor()
    hz = Hz()

    print("Starting force sensor publisher!")

    while True:
        Fxyz = f.read()
        if Fxyz is not None:
            msg = sensor_data_to_force_msg(Fxyz)
            publisher.send(msg.SerializeToString())
            hz.clock()
        time.sleep(0.002)
        print(hz.get_hz())

# seebelow 

Code for the [SeeBelow](https://raghavauppuluri13.github.io/seebelow.github.io/) paper's experimental setup. 

## Compatability
Requires two machines, one connected to franka panda robot running a PREEMPT_RT kernel with the C++ control loop and another connected to force sensor/camera running the python interface

#### Tested Robots
- Franka Panda

#### Tested Sensors
- cameras: D415 (best performance)
- force sensors: [USL-AP Force Sensor](https://tecgihan.co.jp/en/products/forcesensor-amplifier-interface/forcesensor-3-axis/usl-ap_series/), can replace with any other 3axis force sensor

#### Tested OS
- linux: Ubuntu 20.04

#### Tested Python
Highly recommended to use [pyenv](https://github.com/pyenv/pyenv) for virtual environments

- 3.9.18

## Install (todo: make a bash script)

### On both machines
1. Install repo recursively
```
git clone --recursive https://github.com/raghavauppuluri13/seebelow.git
cd seebelow
```

### Setup (Robot/C++ machine)
1. Setup deoxsys 

```
cd external/deoxys_control/deoxys
./InstallPackage
make -j build_franka=1
```

### Setup (Python machine)

1. Setup deoxys

```
cd external/deoxys_control/deoxys
./InstallPackage
make -j build_deoxys=1
pip install -U -r requirements.txt
```

2. Install seebelow 

```
cd ../../../
pip3 install -e .
```

## Hardware

- [panda sensor coupler](https://cad.onshape.com/documents/3bb07ca61024c88e21eb68b6/w/43dd44890b4473c7ef849956/e/c67a8e99ae85981aa134053b)
- [tumors](https://cad.onshape.com/documents/6073bfa153d8c47e7b077db7/w/b9bd611d4deeaa7d92c61b7f/e/801db759efd267626db254a6)

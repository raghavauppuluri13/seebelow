#!/bin/bash

algos=("pets_reacher" "pets_pusher" "planet_cartpole_swingup" "planet_finger_spin" "mbpo_halfcheetah" "mbpo_inv_pendulum")
tumors=("pets_reacher" "pets_pusher" "planet_cartpole_swingup" "planet_finger_spin" "mbpo_half_cheetah_v4" "mbpo_inv_pendulum_v4")
num_palp=("pets" "pets" "planet" "planet" "mbpo" "mbpo")
device=("cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0")
envs=$main_envs
tsp -S 6

for i in ${!envs[@]}; do
    if [[ "${algos[$i]}" == "planet" ]]; then
        tsp python -m mbrl.examples.main algorithm=${algos[$i]} dynamics_model=${algos[$i]} overrides=${envs[$i]} device=${device[$i]}
    else
        tsp python -m mbrl.examples.main algorithm=${algos[$i]} overrides=${envs[$i]} device=${device[$i]}
    fi
done

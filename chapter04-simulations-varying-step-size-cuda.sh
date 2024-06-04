#!/bin/bash

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-3 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-2 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 0.1 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-3 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-2 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 0.1 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-3 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 1e-2 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --step_size 0.1 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-3 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-2 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 0.1 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-3 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-2 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 0.1 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-3 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 1e-2 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --step_size 0.1 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

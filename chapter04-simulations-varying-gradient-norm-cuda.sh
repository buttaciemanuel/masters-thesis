#!/bin/bash

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset femnist --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 10 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm fedavg --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.7 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 1.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --max_gradient_norm 10.0 --output_directory simulations --device cuda --log

python chapter04-train.py --dataset cifar10 --convergence_rounds_threshold 0.30 --n_iterations 50 --n_epochs 10 --algorithm ours --beta 0.5 --weighting_scheme adjacency --class_imbalance 100 --data_imbalance 1 --output_directory simulations --device cuda --log

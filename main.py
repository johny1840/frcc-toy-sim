# Fractal Recursive Computational Cosmology (FRCC) Toy Simulation Core

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import sys
import subprocess
import argparse
from datetime import datetime
from math import log, exp
import csv

# ----- Node Definition -----
class Node:
    def __init__(self, node_id, state=0.5):
        self.id = node_id
        self.state = state
        self.entropy = np.random.rand() * 0.1  # Initial entropy reservoir
        self.neighbors = []

    def update(self, energy_density, depth):
        delta_entropy = compute_entropy_gradient(self, self.neighbors)
        coherence = evaluate_coherence(self.neighbors)
        scale_factor = log(1 + depth + 1e-6)
        curvature = compute_curvature(self.neighbors)
        branching_bias = sigmoid(curvature * scale_factor)

        # Entropy amplification: low coherence enables creative restructuring
        entropy_amplifier = self.entropy * (1 - coherence)
        updated_state = self.state + delta_entropy * entropy_amplifier * branching_bias

        # Update node state and decay entropy reservoir
        self.state = normalize(updated_state)
        self.entropy *= entropy_decay(depth)

# ----- Lattice Definition -----
class Lattice:
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(i, np.random.rand()) for i in range(size)]
        self.graph = nx.Graph()
        self.entropy_history = []
        self.structural_entropy_history = []
        self.connect_nodes_fractal()

    def connect_nodes_fractal(self):
        for i, node in enumerate(self.nodes):
            self.graph.add_node(i)
            for j in range(i+1, self.size):
                distance = abs(i - j + 1)
                if distance == 0:
                    continue
                if np.random.rand() < 1.0 / distance:  # Fractal-like sparse links
                    node.neighbors.append(self.nodes[j])
                    self.nodes[j].neighbors.append(node)
                    self.graph.add_edge(i, j)

    def propagate(self, steps):
        for depth in range(1, steps + 1):
            for node in self.nodes:
                node.update(local_energy(node), depth)
            avg_entropy = np.mean([node.entropy for node in self.nodes])
            self.entropy_history.append(avg_entropy)
            self.structural_entropy_history.append(self.compute_structural_entropy())

    def compute_structural_entropy(self):
        states = np.array([node.state for node in self.nodes])
        hist, _ = np.histogram(states, bins=10, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

# ----- Functional Components -----
def compute_entropy_gradient(node, neighbors):
    if not neighbors:
        return 0
    avg_neighbor_state = np.mean([n.state for n in neighbors])
    return avg_neighbor_state - node.state

def evaluate_coherence(neighbors):
    if not neighbors:
        return 0
    states = np.array([n.state for n in neighbors])
    return 1.0 - np.std(states)  # Lower variance = higher coherence

def compute_curvature(neighbors):
    if not neighbors:
        return 0
    return len(neighbors) - 2  # Simple proxy for branching curvature

def sigmoid(x):
    return 1 / (1 + exp(-x))

def normalize(value):
    return max(0.0, min(1.0, value))

def local_energy(node):
    return 0.5  # Placeholder (can be upgraded to dynamic field later)

def entropy_decay(depth):
    return 1.0 - 0.05 * log(1 + depth)  # Recursive depth reduces entropy gradually

# ----- Visualization -----
def visualize_lattice(lattice, output_dir, title="FRCC Node Network"):
    node_colors = [node.state for node in lattice.nodes]
    pos = nx.spring_layout(lattice.graph)
    fig, ax = plt.subplots()
    nx.draw(lattice.graph, pos, node_color=node_colors, cmap=plt.cm.viridis, with_labels=False, node_size=100, ax=ax)
    plt.title(title)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(node_colors)
    fig.colorbar(sm, ax=ax, label='Node State')
    output_file = os.path.join(output_dir, "frcc_network.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    try:
        if sys.platform == "win32":
            os.startfile(output_file)
        elif sys.platform == "darwin":
            subprocess.run(["open", output_file])
        else:
            subprocess.run(["xdg-open", output_file])
    except Exception as e:
        print("Could not auto-open image:", e)

# ----- Entropy Plotting -----
def plot_entropy_over_time(lattice, output_dir):
    plt.figure()
    steps = range(1, len(lattice.entropy_history) + 1)
    plt.plot(steps, lattice.entropy_history, marker='o', label="Avg Node Entropy (Reservoir)")
    plt.plot(steps, lattice.structural_entropy_history, marker='x', label="Structural Entropy (State Spread)")
    plt.xlabel("Recursion Step")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution Over Time")
    plt.grid(True)
    plt.legend()
    entropy_file = os.path.join(output_dir, "entropy_evolution.png")
    plt.savefig(entropy_file, dpi=300)
    plt.close()
    try:
        if sys.platform == "win32":
            os.startfile(entropy_file)
        elif sys.platform == "darwin":
            subprocess.run(["open", entropy_file])
        else:
            subprocess.run(["xdg-open", entropy_file])
    except Exception as e:
        print("Could not auto-open entropy plot:", e)

# ----- CSV Export -----
def export_entropy_data(lattice, output_dir):
    csv_file = os.path.join(output_dir, "frcc_sim_data.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Node Entropy", "Structural Entropy"])
        for i, (node_e, struct_e) in enumerate(zip(lattice.entropy_history, lattice.structural_entropy_history), 1):
            writer.writerow([i, node_e, struct_e])

# ----- Example Usage -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=50, help="Number of nodes in the lattice")
    parser.add_argument("--steps", type=int, default=10, help="Number of recursion steps")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"frcc_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    lattice = Lattice(size=args.nodes)
    lattice.propagate(steps=args.steps)
    visualize_lattice(lattice, output_dir, f"FRCC Simulation after {args.steps} Steps")
    plot_entropy_over_time(lattice, output_dir)
    export_entropy_data(lattice, output_dir)

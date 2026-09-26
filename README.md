# Genetic Algorithm - Function Optimizer with GUI

A genetic algorithm that minimizes benchmark functions, run through a Tkinter GUI with a live fitness plot.

## At a Glance

- **Stack:** Python, NumPy, Matplotlib, Tkinter
- **Context:** CP 468 Artificial Intelligence term project, Wilfrid Laurier University
- **State:** Complete

## Features

- Selection, crossover, mutation and elitism
- Tunable population size, mutation rate and generation count
- Built-in Sphere, Rosenbrock and Himmelblau functions, plus custom ones
- Live Matplotlib chart of best fitness per generation
- No code needed to run experiments, everything is set in the GUI

## Objective Functions

- **Sphere:** smooth bowl, tests basic convergence to the global minimum
- **Rosenbrock:** narrow curved valley, tests progress on hard landscapes
- **Himmelblau:** four minima, tests behaviour on multimodal functions

## Project Structure

```
artificial-intelligence-genetic-algorithm/
├── genetic_algorithm.py            # GA, objective functions and Tkinter GUI
└── genetic-algorithm-overview.pdf  # Project write-up
```

## Running Locally

1. Clone the repo and move into the project folder:
   ```bash
   git clone https://github.com/nakulpatel0306/artificial-intelligence-genetic-algorithm.git
   cd artificial-intelligence-genetic-algorithm/artificial-intelligence-genetic-algorithm
   ```
2. Install dependencies (Python 3.8+) and launch the GUI:
   ```bash
   pip install numpy matplotlib
   python genetic_algorithm.py
   ```

## Team

Romin Gandhi, Jenish Bharucha, Nakul Patel, Arsh Patel, Dhairya Patel, Paarth Bagga, Devarth Trivedi, Gleb Silin, Emmet Currie, Parker Riches

Built as coursework for CP 468 at Wilfrid Laurier University. Please do not copy for academic submissions.

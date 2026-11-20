# Genetic Algorithm - Function Optimizer With GUI

A genetic algorithm that minimizes benchmark functions. You set it up in a Tkinter GUI, and it plots the best fitness per generation when the run finishes.

## At a Glance

- **Stack:** Python, NumPy, Matplotlib, Tkinter
- **Context:** CP 468 Artificial Intelligence term project, Wilfrid Laurier University
- **State:** Complete

## Features

- Roulette-wheel selection weighted by inverse fitness, single-point crossover, random mutation, and elitism (top 4 carried over)
- Tunable population size, mutation rate and generation count
- Built-in Sphere, Rosenbrock, Himmelblau, N=3 and N=5 functions, plus custom ones typed as a Python expression in `x`
- Matplotlib chart of best fitness per generation, shown at the end of a run
- No code needed to run experiments, everything is set in the GUI

## Objective Functions

- **Sphere:** smooth bowl, tests basic convergence to the global minimum
- **Rosenbrock:** narrow curved valley, tests progress on hard landscapes
- **Himmelblau:** four minima, tests behaviour on multimodal functions (2 variables)
- **N=3 and N=5:** sums of products over 18 and 27 variables

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
   python3 genetic_algorithm.py
   ```
3. Pick a function, enter the number of variables and a `min,max` bound for each one (for example `-5,5`), then press **Start Genetic Algorithm**. Progress prints to the terminal.

## Team

Romin Gandhi, Jenish Bharucha, Nakul Patel, Arsh Patel, Dhairya Patel, Paarth Bagga, Devarth Trivedi, Gleb Silin, Emmet Currie, Parker Riches

Built as coursework for CP 468 at Wilfrid Laurier University. Please do not copy for academic submissions.

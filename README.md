# Go-Explore Phase 1 Implementation

This repository contains a Jupyter Notebook implementing **Phase 1 ("Explore Until Solved")** of the Go-Explore algorithm for a graduate-level reinforcement learning midterm project.

## Paper Reference

**Go-Explore: A New Approach for Hard-Exploration Problems**  
Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, and Jeff Clune  
Uber AI Labs (2019)

Paper link: https://huggingface.co/papers/1901.10995

## Overview

Go-Explore is a novel exploration algorithm designed to solve "hard-exploration" problems in reinforcement learning.

This implementation demonstrates Phase 1 of Go-Explore using the FrozenLake-v1 environment from Gymnasium.

## Requirements

-   Python >= 3.9
-   gymnasium
-   numpy
-   matplotlib

Install dependencies:

```bash
pip install gymnasium numpy matplotlib
```

## Usage

Simply open and run the Jupyter Notebook:

```bash
jupyter notebook go_explore_phase1.ipynb
```

Or use JupyterLab:

```bash
jupyter lab go_explore_phase1.ipynb
```

Run all cells in order to:

1. Learn about Go-Explore's core concepts
2. See the Phase 1 algorithm in action
3. Visualize exploration progress
4. Compare with random exploration baseline

## What's Included

The notebook contains:

1. **Introduction**: Overview of Go-Explore and its motivation
2. **Algorithm Explanation**: Detailed description of Phase 1 components
3. **Implementation**: Complete working code with clear comments
4. **Visualization**: Plots showing exploration progress and discovered cells
5. **Analysis**: Archive statistics and top trajectories
6. **Comparison**: Go-Explore vs pure random exploration
7. **Conclusion**: Summary of findings and discussion of Phase 2

## Key Concepts Demonstrated

-   **Archive**: Storage of visited states and trajectories
-   **State Abstraction**: Converting states to abstract "cells"
-   **Return-Then-Explore**: Systematically returning to promising states
-   **Deterministic Replay**: Reliable trajectory execution in deterministic environments

## Limitations

This implementation focuses on Phase 1 only:

-   Works in deterministic environments only
-   No robustification (Phase 2)
-   Simple state abstraction
-   Random cell selection (no prioritization heuristics)

Phase 2 would involve training a robust policy via imitation learning to handle stochastic environments.

## Expected Results

When run on FrozenLake-v1 (8x8, deterministic), Go-Explore should:

-   Find the goal state (reward = 1.0)
-   Solve the problem more efficiently than random exploration
-   Demonstrate systematic exploration vs random wandering

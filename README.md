# Go-Explore: Complete Two-Phase Implementation

This repository contains a Jupyter Notebook implementing **both phases** of the Go-Explore algorithm for a graduate-level reinforcement learning project:

-   **Phase 1**: "Explore Until Solved" (archive-based exploration)
-   **Phase 2**: "Robustification" (PPO with Backward Algorithm)

## Paper Reference

**Go-Explore: A New Approach for Hard-Exploration Problems**  
Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, and Jeff Clune  
Uber AI Labs (2019)

Paper link: https://huggingface.co/papers/1901.10995

## Overview

Go-Explore is a novel exploration algorithm designed to solve "hard-exploration" problems in reinforcement learning.

This implementation demonstrates **both phases** of Go-Explore using a custom 8×8 FrozenLake environment with the goal in the bottom-right quadrant:

-   **Phase 1** systematically explores using an archive with **weighted cell selection** to discover solution trajectories
-   **Phase 2** trains a robust neural network policy using PPO and the Backward Algorithm curriculum

**Key Features:**

-   ✅ Weighted cell selection (prioritizes frontier exploration)
-   ✅ Comprehensive visualization of archive statistics and selection patterns
-   ✅ Custom 10% slipperiness testing (validates Phase 2 generalization)
-   ✅ Complete two-phase implementation matching the paper's methodology

## Requirements

-   Python >= 3.9
-   gymnasium
-   numpy
-   matplotlib
-   torch (PyTorch)

Install dependencies:

```bash
pip install gymnasium numpy matplotlib torch
```

## Usage

Simply open and run the Jupyter Notebook:

```bash
jupyter notebook go_explore_mv.ipynb
```

Or use JupyterLab:

```bash
jupyter lab go_explore_mv.ipynb
```

Run all cells in order to:

1. Learn about Go-Explore's core concepts
2. See Phase 1 (exploration) in action
3. Visualize Phase 1 exploration progress and compare with random exploration
4. Train a robust policy in Phase 2 using PPO and Backward Algorithm
5. Evaluate the final trained policy

## What's Included

The notebook contains:

### Phase 1 — Exploration

1. **Introduction**: Overview of Go-Explore and its motivation
2. **Algorithm Explanation**: Detailed description of Phase 1 components (including weighted selection)
3. **Implementation**: Archive-based exploration with weighted cell selection
4. **Visualization**: Plots showing exploration progress, cell selection patterns, and archive statistics
5. **Analysis**: Archive statistics, visit distribution, and top trajectories
6. **Comparison**: Go-Explore vs pure random exploration

### Phase 2 — Robustification

7. **Neural Network Architecture**: MLP-based Actor-Critic policy
8. **PPO Implementation**: Full PPO training loop with GAE
9. **Backward Algorithm**: Curriculum learning from goal to start
10. **Training Visualization**: Success rates and losses across curriculum stages
11. **Final Evaluation**: Policy performance over 100 test episodes
12. **Comparison**: Phase 1 trajectory vs Phase 2 learned policy
13. **Custom 10% Slipperiness Test**: Evaluates policy on manageable stochastic environment to validate generalization

## Key Concepts Demonstrated

### Phase 1 Concepts

-   **Archive**: Storage of visited states and trajectories
-   **State Abstraction**: Converting states to abstract "cells"
-   **Weighted Cell Selection**: Prioritizes rarely-visited frontier cells (✅ implemented)
-   **Return-Then-Explore**: Systematically returning to promising states
-   **Deterministic Replay**: Reliable trajectory execution in deterministic environments

### Phase 2 Concepts

-   **Backward Algorithm**: Curriculum learning starting from near the goal
-   **PPO (Proximal Policy Optimization)**: Policy gradient method with clipped objective
-   **Actor-Critic Architecture**: Separate policy and value function heads
-   **GAE (Generalized Advantage Estimation)**: Variance reduction for advantage computation
-   **Robustification**: Converting brittle trajectories into robust neural network policies
-   **Stochastic Generalization**: Testing policy transfer from deterministic to 10% slipperiness environments

## Implementation Notes

This implementation demonstrates both phases of Go-Explore with some simplifications for educational clarity:

### Compared to the Original Paper (Atari):

-   **Environment**: FrozenLake (discrete states) instead of Atari (pixel observations)
-   **Network**: MLP instead of CNN (appropriate for discrete states)
-   **State representation**: One-hot encoding instead of image downsampling
-   **Training scale**: Fewer episodes (simpler environment requires less training)

### What Remains Faithful to Go-Explore:

✅ Two-phase approach (exploration → robustification)  
✅ Archive-based systematic exploration  
✅ Weighted cell selection (prioritizes frontier cells)  
✅ PPO for policy optimization  
✅ Backward Algorithm curriculum learning  
✅ Trajectory-to-policy conversion methodology  
✅ Custom 10% slipperiness testing (deterministic training → manageable stochastic evaluation)

## Expected Results

### Phase 1 Results

When run on FrozenLake-v1 (8×8, deterministic), Phase 1 should:

-   Systematically discover all reachable cells
-   Find the goal state (reward = 1.0)
-   Solve the problem more efficiently than random exploration
-   Store a trajectory to the goal in the archive

### Phase 2 Results

After training with the Backward Algorithm, Phase 2 should:

-   **Deterministic Environment**: Achieve >90% success rate (typically 95-100%)
-   Learn an **optimal or near-optimal policy** (14-step Manhattan distance solution)
-   Successfully progress through all curriculum stages
-   Convert the brittle Phase 1 trajectory into a robust neural network policy

### Custom 10% Slipperiness Test Results

After training on deterministic environment, testing on custom 10% slipperiness environment:

-   **Expected success rate**: 80-95%
-   **Action noise**: 10% chance of unintended movement per action
-   **Interpretation**: >80% success demonstrates successful robustification and generalization
-   **Key validation**: Proves policy learned navigation skills, not just trajectory memorization
-   **Advantage**: Much more manageable than standard 67% slipperiness, allowing proper testing

### Typical Training Pattern

-   **Stage 1-2**: Quick convergence (1-5 iterations) near the goal
-   **Later stages**: Initial struggle followed by breakthrough to 100% success
-   **Final policy**: Consistent, deterministic behavior achieving the goal every episode

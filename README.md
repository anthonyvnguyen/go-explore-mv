# Enhanced Go-Explore: Dyna, Sweeping & Learned Selection

This repository contains a Jupyter Notebook implementing an **enhanced version** of the Go-Explore algorithm for a graduate-level reinforcement learning project. It builds upon the standard two-phase Go-Explore implementation by integrating classic reinforcement learning techniques to investigate potential improvements in exploration efficiency.

-   **File**: `go_explore_final.ipynb`
-   **Environment**: Custom 16×16 FrozenLake with goal in the bottom-right quadrant

## Paper References

**Base Algorithm:**

-   **Go-Explore**: Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J. (2019). _Go-Explore: A New Approach for Hard-Exploration Problems_. [arXiv:1901.10995](https://huggingface.co/papers/1901.10995)

**Enhancements Implemented:**

-   **Dyna (Model-Based Planning)**: Sutton, R. S. (1990). _Integrated architectures for learning, planning, and reacting_.
-   **Prioritized Sweeping**: Moore, A. W., & Atkeson, C. G. (1993). _Prioritized sweeping: Reinforcement learning with less data and less time_.
-   **Learned Cell Selection (REINFORCE)**: Williams, R. J. (1992). _Simple statistical gradient-following algorithms for connectionist reinforcement learning_.
-   **Near-Miss Integration**: Inspired by Salimans et al. (2018). _Learning Montezuma’s Revenge from a Single Demonstration_.

## Overview

This implementation demonstrates **both phases** of Go-Explore, augmented with additional mechanisms:

1.  **Phase 1 (Enhanced Exploration)**:

    -   **Dyna**: Learns a transition model to perform "imagined" rollouts, estimating cell novelty without physical visits.
    -   **Prioritized Sweeping**: Propagates value changes backwards to prioritize exploration from promising areas.
    -   **Learned Cell Selection**: Replaces heuristic weighting with a policy network trained via REINFORCE to select cells based on features like visit count and novelty.

2.  **Phase 2 (Robustification)**:
    -   **Backward Algorithm**: Curriculum learning starting from near the goal.
    -   **Near-Miss Curriculum**: Integrates "near-miss" states (states close to the goal but not quite there) into the curriculum to improve robustness.
    -   **PPO Training**: Trains a robust neural network policy.

## Environment Details

This implementation provides 5 different 16×16 map layouts to test exploration under varying conditions:

1.  **Original**: Strategic map with barriers and holes (Goal at 11,11).
2.  **FourRooms**: 4 rooms connected by narrow doorways (Goal at 15,15).
3.  **Bottleneck**: Two large areas separated by a single narrow passage (Goal at 15,15).
4.  **Maze**: A generated maze with a single solution path (Goal at 15,15).
5.  **Open**: Mostly open space with minimal obstacles (Goal at 15,15).

The default environment uses the **Original** map.

## Key Features

### Phase 1 Enhancements

-   ✅ **Dyna-Q Integration**: Updates internal models and performs planning steps.
-   ✅ **Prioritized Sweeping**: Maintains a priority queue for value propagation.
-   ✅ **Learned Selection Policy**: Neural network selector (Inputs: cell depth, visits, novelty, priority).
-   ✅ **Comparison Framework**: Benchmarks "Enhanced" vs. "Standard" Go-Explore.

### Phase 2 Features

-   ✅ **Behavior Cloning Warm-Start**: Accelerates initial policy learning.
-   ✅ **Backward Algorithm with Near-Misses**: Curriculum includes high-value non-goal states.
-   ✅ **PPO (Proximal Policy Optimization)**: Robust policy training.
-   ✅ **Stochastic Robustness Test**: Validates policy generalization on a custom 10% slippery environment.

## Experimental Findings

**Note on Performance**: As detailed in the notebook, experimental results indicate that **these enhancements do not consistently improve performance** on the FrozenLake environment compared to the standard Go-Explore implementation.

-   The computational overhead of the enhancements often outweighs their benefits in this relatively simple domain.
-   Standard Go-Explore's simple heuristic (`weight ∝ 1/√visits`) is remarkably effective and hard to beat for this specific problem.
-   This serves as an important educational result: complex algorithms are not always better for every problem instance.

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

Open and run the final notebook:

```bash
jupyter notebook go_explore_final.ipynb
```

Run all cells to see:

1.  Implementation of the enhanced components.
2.  Comparative experiments between Enhanced and Standard Phase 1.
3.  Phase 2 training with the Backward Algorithm and Near-Miss curriculum.
4.  Final policy evaluation.

## What's Included in the Notebook

1.  **Enhanced Phase 1 Implementation**:

    -   `DynaModel`: Learned transition model ($s, a \to s', r$).
    -   `PrioritizedSweeping`: Value iteration with priority queues.
    -   `SelectionPolicy`: REINFORCE-based neural selector.
    -   `EnhancedArchive`: Integration of all components.

2.  **Standard Phase 1 Implementation**:

    -   Baseline for comparison.

3.  **Phase 2 Implementation**:

    -   Actor-Critic Network.
    -   PPO Training Loop.
    -   Backward Algorithm with Near-Miss integration.

4.  **Analysis & Visualization**:
    -   Comparisons of exploration efficiency.
    -   Training curves for policy robustification.
    -   Robustness tests on stochastic environments.

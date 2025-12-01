import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import heapq
import random
from collections import defaultdict, deque
import math
import time
import itertools

# ============================================================================
# 1. Core Functions (Recreated from Notebook)
# ============================================================================

class MapGenerator:
    @staticmethod
    def get_empty_grid(size=16):
        return [['F' for _ in range(size)] for _ in range(size)]

    @staticmethod
    def create_original():
        """Original strategic map"""
        custom_map = []
        for i in range(16):
            row = ['F'] * 16
            custom_map.append(row)
        custom_map[0][0] = 'S'
        custom_map[11][11] = 'G'
        
        holes = [
            (4, 4), (4, 5), (4, 6), (4, 7),
            (8, 8), (8, 9), (8, 10), (8, 11),
            (12, 6), (12, 7), (12, 8), (12, 9),
            (6, 11), (7, 11), (8, 11), (9, 11), (10, 11),
            (10, 2), (10, 3), (10, 4), (10, 5),
            (2, 10), (3, 10), (4, 10), (5, 10),
            (1, 5), (1, 8), (1, 12),
            (5, 1), (5, 9), (5, 14),
            (9, 1), (9, 6), (9, 13),
            (13, 3), (13, 8), (13, 12),
            (14, 1), (14, 5), (14, 10),
        ]
        for r, c in holes:
            custom_map[r][c] = 'H'
        return [''.join(row) for row in custom_map]

    @staticmethod
    def create_four_rooms():
        grid = [['F' for _ in range(16)] for _ in range(16)]
        # Walls
        for i in range(16):
            grid[7][i] = 'H'  # Horizontal divider
            grid[i][7] = 'H'  # Vertical divider
            
        # Doorways
        grid[7][3] = 'F'  # Top-left to Bottom-left
        grid[7][12] = 'F' # Top-right to Bottom-right
        grid[3][7] = 'F'  # Top-left to Top-right
        grid[12][7] = 'F' # Bottom-left to Bottom-right
        
        grid[0][0] = 'S'
        grid[15][15] = 'G'
        return [''.join(row) for row in grid]

    @staticmethod
    def create_bottleneck():
        grid = [['F' for _ in range(16)] for _ in range(16)]
        # Wall in the middle column
        for i in range(16):
            if i != 8: # Gap at row 8
                grid[i][8] = 'H'
        
        grid[0][0] = 'S'
        grid[15][15] = 'G'
        return [''.join(row) for row in grid]

    @staticmethod
    def create_maze():
        grid = [['F' for _ in range(16)] for _ in range(16)]
        # Simple snake pattern
        for row in range(1, 15, 2):
            for col in range(1, 15):
                if (row // 2) % 2 == 0:
                    if col < 14: grid[row][col] = 'H'
                else:
                    if col > 1: grid[row][col] = 'H'
                    
        grid[0][0] = 'S'
        grid[15][15] = 'G'
        return [''.join(row) for row in grid]
    
    @staticmethod
    def create_open():
        grid = [['F' for _ in range(16)] for _ in range(16)]
        grid[0][0] = 'S'
        grid[15][15] = 'G'
        # Few random holes
        grid[5][5] = 'H'
        grid[10][10] = 'H'
        return [''.join(row) for row in grid]

maps = {
    "Original": MapGenerator.create_original(),
    "FourRooms": MapGenerator.create_four_rooms(),
    "Bottleneck": MapGenerator.create_bottleneck(),
    "Maze": MapGenerator.create_maze(),
    "Open": MapGenerator.create_open()
}

def get_cell(state):
    """State abstraction function: converts raw state to a cell representation."""
    return state

def rollout_to_cell(env, trajectory):
    """Deterministically return to a cell by executing the stored trajectory."""
    state, info = env.reset()
    total_reward = 0
    terminated = False
    
    for action in trajectory:
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    return state, total_reward, terminated

def explore_from_cell_original(env, trajectory, k_steps, stickiness=0.9):
    """
    Original exploration function for reference.
    """
    new_cells = {}
    
    # Return to the starting cell
    state, reward_so_far, terminated = rollout_to_cell(env, trajectory)
    
    if terminated:
        return new_cells
    
    current_trajectory = trajectory.copy()
    last_action = None
    
    for _ in range(k_steps):
        if last_action is not None and random.random() < stickiness:
            action = last_action
        else:
            action = env.action_space.sample()
        
        state, reward, terminated, truncated, info = env.step(action)
        current_trajectory.append(action)
        reward_so_far += reward
        last_action = action
        
        cell = get_cell(state)
        new_cells[cell] = (current_trajectory.copy(), reward_so_far)
        
        if terminated or truncated:
            break
    
    return new_cells

def go_explore_phase1(env, max_iterations=1000, k_explore=10, target_reward=1.0, 
                     use_weighted_selection=True, stickiness=0.9):
    """
    Original Go-Explore Phase 1 algorithm from baseline.
    Recreated here for standalone script execution.
    """
    initial_state, _ = env.reset()
    initial_cell = get_cell(initial_state)
    
    archive = {
        initial_cell: {
            'trajectory': [],
            'reward': 0.0,
            'times_chosen': 0,
            'times_visited': 0,
            'first_visit': 0
        }
    }
    
    history = {
        'iterations': [],
        'cells_discovered': [],
        'max_reward': [],
        'solved_iteration': None
    }
    
    solved = False
    
    for iteration in range(max_iterations):
        # Select cell
        cells = list(archive.keys())
        if use_weighted_selection:
            weights = [(1.0 / (archive[c]['times_chosen'] + 0.1) ** 0.5) for c in cells]
            total = sum(weights)
            weights = [w / total for w in weights]
            cell = random.choices(cells, weights=weights, k=1)[0]
        else:
            cell = random.choice(cells)
        
        archive[cell]['times_chosen'] += 1
        trajectory = archive[cell]['trajectory']
        
        # Explore
        new_cells = explore_from_cell_original(env, trajectory, k_explore, stickiness)
        
        # Update archive
        for new_cell, (new_trajectory, new_reward) in new_cells.items():
            if new_cell in archive:
                archive[new_cell]['times_visited'] += 1
            
            should_update = (new_cell not in archive or 
                           new_reward > archive[new_cell]['reward'] or
                           (new_reward == archive[new_cell]['reward'] and 
                            len(new_trajectory) < len(archive[new_cell]['trajectory'])))
            
            if should_update:
                if new_cell not in archive:
                    archive[new_cell] = {
                        'trajectory': new_trajectory,
                        'reward': new_reward,
                        'times_chosen': 0,
                        'times_visited': 1,
                        'first_visit': iteration
                    }
                else:
                    archive[new_cell]['trajectory'] = new_trajectory
                    archive[new_cell]['reward'] = new_reward
                
                if new_reward >= target_reward and not solved:
                    solved = True
                    history['solved_iteration'] = iteration
        
        history['iterations'].append(iteration)
        history['cells_discovered'].append(len(archive))
        history['max_reward'].append(max(cell_data['reward'] for cell_data in archive.values()))
    
    return archive, history


# ============================================================================
# 2. Dyna Component (Model-Based Planning)
# ============================================================================

class DynaModel:
    def __init__(self, num_states=256, num_actions=4):
        # Transition counts: [state, action, next_state]
        self.transition_counts = np.zeros((num_states, num_actions, num_states), dtype=np.int32)
        # Total transitions from state-action: [state, action]
        self.transition_counts_total = np.zeros((num_states, num_actions), dtype=np.int32)
        # Reward estimates: [state, action] (running average)
        self.reward_estimates = np.zeros((num_states, num_actions), dtype=np.float32)
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Track which states have been visited/observed
        self.known_states = set()

    def update(self, state, action, next_state, reward):
        """Update transition and reward models from real experience."""
        self.transition_counts[state, action, next_state] += 1
        self.transition_counts_total[state, action] += 1
        
        # Update reward estimate (incremental mean)
        n = self.transition_counts_total[state, action]
        current_est = self.reward_estimates[state, action]
        self.reward_estimates[state, action] = current_est + (reward - current_est) / n
        
        self.known_states.add(state)
        self.known_states.add(next_state)

    def predict(self, state, action):
        """
        Sample next state and reward from learned model.
        Returns: next_state, reward
        """
        if self.transition_counts_total[state, action] == 0:
            # If no experience, return self and 0 reward (or could be random)
            return state, 0.0
        
        # Sample next state proportional to counts
        probs = self.transition_counts[state, action] / self.transition_counts_total[state, action]
        next_state = np.random.choice(self.num_states, p=probs)
        
        reward = self.reward_estimates[state, action]
        return next_state, reward
        
    def get_novelty(self, state, action):
        """
        Heuristic for novelty/uncertainty of a state-action pair.
        Inverse square root of visit count.
        """
        count = self.transition_counts_total[state, action]
        return 1.0 / np.sqrt(count + 0.1)

def dyna_planning(dyna_model, archive, start_cell, num_rollouts=16, depth=6):
    """
    Perform short imagined rollouts from a cell using the learned model.
    Returns estimated novelty (expected new cells per step).
    """
    if start_cell not in dyna_model.known_states:
        return 0.0
        
    total_new_cells = 0
    total_steps = 0
    
    # Gather all currently known cells in archive for novelty check
    archive_cells = set(archive.keys())
    
    for _ in range(num_rollouts):
        current_state = start_cell
        trajectory_novelty = 0
        
        for _ in range(depth):
            # Simple policy for imagination: random
            action = random.randint(0, dyna_model.num_actions - 1)
            
            next_state, _ = dyna_model.predict(current_state, action)
            
            # Check if we found something "new" (not in archive)
            # This is a proxy for exploration potential
            if next_state not in archive_cells:
                trajectory_novelty += 1
                
            current_state = next_state
            total_steps += 1
            
            # If terminal (model might not explicitly track terminal, but we can infer if self-loop with 0 reward in some envs, 
            # but here we just run for fixed depth unless stuck)
            
        total_new_cells += trajectory_novelty
        
    if total_steps == 0:
        return 0.0
        
    return total_new_cells / total_steps

# ============================================================================
# 3. Prioritized Sweeping
# ============================================================================

class PrioritizedSweeping:
    def __init__(self, num_states=256, gamma=0.99, threshold=0.01):
        self.V = np.zeros(num_states, dtype=np.float32)
        self.gamma = gamma
        self.threshold = threshold
        
        # Priority queue: stores (-priority, state) because heapq is min-heap
        self.pq = [] 
        # Keep track of what's in queue to update priorities
        self.entry_finder = {} 
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
        
        # Predecessors: map next_state -> list of (state, action)
        # Used for reverse propagation
        self.predecessors = defaultdict(set)

    def add_predecessor(self, state, action, next_state):
        self.predecessors[next_state].add((state, action))

    def update_priority(self, state, priority):
        """Add or update the priority of a state."""
        if state in self.entry_finder:
            self.remove_task(state)
        count = next(self.counter)
        entry = [-priority, count, state]
        self.entry_finder[state] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, state):
        """Mark an existing task as removed."""
        entry = self.entry_finder.pop(state)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, state = heapq.heappop(self.pq)
            if state is not self.REMOVED:
                del self.entry_finder[state]
                return -priority, state
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return not bool(self.entry_finder)

    def update_value(self, state, reward, next_state):
        """
        Perform a TD update for a single transition and return TD error.
        V(s) = V(s) + alpha * (r + gamma * V(s') - V(s))
        But for Prioritized Sweeping, we often just calculate the 'target' and error.
        Standard PS: error = |r + gamma * max_a Q(s', a) - Q(s, a)|
        Here we are using V-values.
        error = |r + gamma * V(s') - V(s)|
        """
        # Simple TD(0) style error for V-values
        target = reward + self.gamma * self.V[next_state]
        error = abs(target - self.V[state])
        
        # We might want to actually update the value here or during the sweep
        # Standard PS updates the value immediately when processing the queue
        # But we also update from real experience.
        # Let's update V with a learning rate
        alpha = 0.1
        self.V[state] += alpha * (target - self.V[state])
        
        return error

    def sweep(self, dyna_model, max_updates=10):
        """
        Process high-priority states from the queue.
        """
        updates = 0
        while not self.is_empty() and updates < max_updates:
            priority, state = self.pop_task()
            
            if priority < self.threshold:
                break
                
            # For V-value sweeping, we conceptually update V(state) based on all possible actions
            # But since we don't have Q-values, we can just re-estimate V(state) from model
            # V(s) = max_a \sum_s' P(s'|s,a) [R(s,a,s') + gamma * V(s')]
            
            max_val = -float('inf')
            for action in range(dyna_model.num_actions):
                # Get expected next state and reward from model
                if dyna_model.transition_counts_total[state, action] > 0:
                    # Expected value
                    # This is computationally expensive if we iterate all next states
                    # Approximation: sample or just use the "most likely" or average
                    # Let's use the weighted average over observed transitions
                    
                    total_trans = dyna_model.transition_counts_total[state, action]
                    counts = dyna_model.transition_counts[state, action]
                    
                    # indices where counts > 0
                    next_states = np.where(counts > 0)[0]
                    
                    expected_return = 0
                    for ns in next_states:
                        prob = counts[ns] / total_trans
                        r = dyna_model.reward_estimates[state, action] # This is avg reward
                        expected_return += prob * (r + self.gamma * self.V[ns])
                    
                    if expected_return > max_val:
                        max_val = expected_return
            
            if max_val > -float('inf'):
                # Update value
                self.V[state] = max_val
                
                # Propagate to predecessors
                for pred_state, pred_action in self.predecessors[state]:
                    # Calculate error for predecessor
                    r = dyna_model.reward_estimates[pred_state, pred_action]
                    error = abs(r + self.gamma * self.V[state] - self.V[pred_state])
                    if error > self.threshold:
                        self.update_priority(pred_state, error)
            
            updates += 1
            
# ============================================================================
# 4. Learned Cell Selector (REINFORCE Policy)
# ============================================================================

class CellSelectorPolicy(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super(CellSelectorPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output score for a single cell
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def extract_cell_features(cell, archive, iteration, dyna_novelty, sweeping_priority):
    """
    Compute features for a cell.
    Features: [depth, time_since_expansion, visit_count, dyna_novelty, sweeping_priority]
    """
    data = archive[cell]
    
    # Normalize features roughly to [0, 1] or [-1, 1]
    depth = len(data['trajectory']) / 20.0 # normalization factor
    time_since = (iteration - data.get('last_chosen', 0)) / 100.0
    visits = data['times_visited'] / 10.0
    novelty = dyna_novelty * 5.0
    priority = sweeping_priority * 5.0
    
    return torch.tensor([depth, time_since, visits, novelty, priority], dtype=torch.float32)

def select_cell_learned(archive, policy, iteration, dyna_estimates, priorities, heuristic_mix=0.5):
    """
    Select a cell using a mixture of learned policy and heuristic.
    """
    cells = list(archive.keys())
    if not cells:
        return None, None, None

    # 1. Learned Scores
    features_list = []
    for cell in cells:
        novelty = dyna_estimates.get(cell, 0.0)
        priority = priorities.get(cell, 0.0)
        feat = extract_cell_features(cell, archive, iteration, novelty, priority)
        features_list.append(feat)
    
    features_tensor = torch.stack(features_list)
    with torch.no_grad():
        learned_scores = policy(features_tensor).squeeze(-1) # (num_cells,)
        learned_probs = F.softmax(learned_scores, dim=0).numpy()
        
    # 2. Heuristic Scores (original Go-Explore weight)
    # Weight ∝ 1 / (times_chosen + 0.1)^0.5
    heuristic_weights = np.array([(1.0 / (archive[c]['times_chosen'] + 0.1) ** 0.5) for c in cells])
    heuristic_probs = heuristic_weights / np.sum(heuristic_weights)
    
    # 3. Mixture
    mixed_probs = (1 - heuristic_mix) * learned_probs + heuristic_mix * heuristic_probs
    mixed_probs = mixed_probs / np.sum(mixed_probs) # ensure sum to 1
    
    # Sample
    idx = np.random.choice(len(cells), p=mixed_probs)
    selected_cell = cells[idx]
    
    # Return data needed for REINFORCE update
    # We need the log_prob of the selected action according to the POLICY (not mixture)
    # But since we sample from mixture, importance sampling? 
    # Or just treat the decision as coming from the policy for the gradient part?
    # The plan implies direct REINFORCE on the selector. 
    # Let's return the log_prob of the selected cell under the learned policy distribution.
    
    log_prob = torch.log_softmax(policy(features_tensor).squeeze(-1), dim=0)[idx]
    
    return selected_cell, log_prob, features_tensor[idx]

def update_cell_selector_policy(policy, optimizer, batch_log_probs, batch_returns, baseline=None):
    """
    Update cell selector policy using REINFORCE.
    """
    if not batch_log_probs:
        return 0.0, baseline

    log_probs = torch.stack(batch_log_probs)
    returns = torch.tensor(batch_returns, dtype=torch.float32)
    
    # Update baseline (moving average)
    if baseline is None:
        baseline = returns.mean().item()
    else:
        baseline = 0.9 * baseline + 0.1 * returns.mean().item()
    
    # Advantage = Return - Baseline
    advantages = returns - baseline
    
    # Loss = -mean(log_prob * advantage)
    loss = -(log_probs * advantages).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), baseline

# ============================================================================
# 5. Enhanced Exploration
# ============================================================================

def explore_from_cell_enhanced(env, trajectory, k_steps, dyna_model, sweeping=None, stickiness=0.9):
    """
    Enhanced exploration using Dyna model to bias first action.
    """
    new_cells = {}
    
    # Return to cell
    state, reward_so_far, terminated = rollout_to_cell(env, trajectory)
    if terminated:
        return new_cells, 0, 0.0

    current_trajectory = trajectory.copy()
    last_action = None
    
    # Bias first action using Dyna novelty
    # Check which action has highest novelty/uncertainty
    best_action = None
    max_novelty = -1.0
    
    if dyna_model:
        for a in range(env.action_space.n):
            nov = dyna_model.get_novelty(state, a)
            if nov > max_novelty:
                max_novelty = nov
                best_action = a
    
    # Epsilon-greedy for the "best" action suggested by novelty
    if best_action is not None and random.random() < 0.3: # 30% chance to follow novelty
        first_action = best_action
    else:
        first_action = env.action_space.sample()

    reward_increase = 0.0
    cells_discovered_count = 0
    
    for i in range(k_steps):
        if i == 0:
            action = first_action
        elif last_action is not None and random.random() < stickiness:
            action = last_action
        else:
            action = env.action_space.sample()
            
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Update Dyna Model with real experience
        if dyna_model:
            dyna_model.update(state, action, next_state, reward)
            
        # Update Prioritized Sweeping with real experience
        if sweeping:
            error = sweeping.update_value(state, reward, next_state)
            sweeping.add_predecessor(state, action, next_state)
            if error > sweeping.threshold:
                sweeping.update_priority(state, error)
            
        current_trajectory.append(action)
        reward_so_far += reward
        
        if reward > 0:
             reward_increase += reward
             
        cell = get_cell(next_state)
        new_cells[cell] = (current_trajectory.copy(), reward_so_far)
        cells_discovered_count += 1
        
        state = next_state
        last_action = action
        
        if terminated or truncated:
            break
            
    return new_cells, cells_discovered_count, reward_increase

# ============================================================================
# 6. Enhanced Main Algorithm
# ============================================================================

def go_explore_phase1_enhanced(env, max_iterations=1000, k_explore=10, target_reward=1.0, 
                              use_dyna=True, use_sweeping=True, use_learned_selector=True,
                              stickiness=0.9):
    """
    Enhanced Go-Explore Phase 1 with Dyna, Prioritized Sweeping, and Learned Selector.
    """
    # Initialize
    initial_state, _ = env.reset()
    initial_cell = get_cell(initial_state)
    
    archive = {
        initial_cell: {
            'trajectory': [],
            'reward': 0.0,
            'times_chosen': 0,
            'times_visited': 0,
            'first_visit': 0,
            'last_chosen': 0
        }
    }
    
    # Components
    dyna_model = DynaModel() if use_dyna else None
    sweeping = PrioritizedSweeping() if use_sweeping else None
    dyna_novelty_cache = {} # Cache for novelty values
    
    # Selector Policy
    selector_policy = CellSelectorPolicy()
    selector_optimizer = optim.Adam(selector_policy.parameters(), lr=1e-3)
    selector_baseline = None
    selector_batch_log_probs = []
    selector_batch_returns = []
    
    history = {
        'iterations': [],
        'cells_discovered': [],
        'max_reward': [],
        'solved_iteration': None
    }
    
    solved = False
    
    # Tracking for rewards/returns for selector training
    
    for iteration in range(max_iterations):
        # 1. Select Cell
        if use_learned_selector:
            # Get priorities from sweeping if available
            # Entry is [-priority, count, state], we want state -> priority
            priorities = {entry[2]: -entry[0] for entry in sweeping.entry_finder.values()} if sweeping else {}
            
            # Use cached Dyna novelty estimates
            dyna_estimates = {c: dyna_novelty_cache.get(c, 0.0) for c in archive.keys()}
            
            cell, log_prob, _ = select_cell_learned(archive, selector_policy, iteration, 
                                                   dyna_estimates, priorities)
        else:
            # Fallback to simple weighted
            cells = list(archive.keys())
            weights = [(1.0 / (archive[c]['times_chosen'] + 0.1) ** 0.5) for c in cells]
            total = sum(weights)
            weights = [w / total for w in weights]
            cell = random.choices(cells, weights=weights, k=1)[0]
            log_prob = None

        archive[cell]['times_chosen'] += 1
        archive[cell]['last_chosen'] = iteration
        trajectory = archive[cell]['trajectory']
        
        # 2. Return to Cell
        state, _, terminated = rollout_to_cell(env, trajectory)
        
        if terminated:
            continue
            
        # 3. Dyna Planning (Imagination)
        if use_dyna:
            novelty = dyna_planning(dyna_model, archive, get_cell(state))
            # Update cache
            dyna_novelty_cache[get_cell(state)] = novelty
            
        # 4. Exploration
        new_cells_data, cells_found, reward_inc = explore_from_cell_enhanced(
            env, trajectory, k_explore, dyna_model, sweeping=sweeping, stickiness=stickiness
        )
        
        # 5. Archive Update
        for new_cell, (new_traj, new_reward) in new_cells_data.items():
            if new_cell in archive:
                archive[new_cell]['times_visited'] += 1
                
            should_update = (new_cell not in archive or 
                           new_reward > archive[new_cell]['reward'] or
                           (new_reward == archive[new_cell]['reward'] and 
                            len(new_traj) < len(archive[new_cell]['trajectory'])))
            
            if should_update:
                if new_cell not in archive:
                    archive[new_cell] = {
                        'trajectory': new_traj,
                        'reward': new_reward,
                        'times_chosen': 0,
                        'times_visited': 1,
                        'first_visit': iteration,
                        'last_chosen': 0
                    }
                    # Add to sweeping queue with high priority?
                    if use_sweeping:
                        sweeping.update_priority(new_cell, 1.0) # High priority for new cells
                else:
                    archive[new_cell]['trajectory'] = new_traj
                    archive[new_cell]['reward'] = new_reward
                
                if new_reward >= target_reward and not solved:
                    solved = True
                    history['solved_iteration'] = iteration

        # 6. Update Models & Priorities (Sweeping)
        # Dyna model was updated inside explore_from_cell_enhanced via calls to update()
        if use_sweeping and use_dyna:
            # Sweeping uses the model to update V-values
            sweeping.sweep(dyna_model)
            
        # 7. Train Selector
        if use_learned_selector and log_prob is not None:
            # Compute return for this selection
            # Simple reward: number of new cells found + reward increase
            step_return = cells_found + reward_inc * 10.0
            
            selector_batch_log_probs.append(log_prob)
            selector_batch_returns.append(step_return)
            
            # Update every 10 iterations (batch)
            if len(selector_batch_log_probs) >= 10:
                loss, selector_baseline = update_cell_selector_policy(
                    selector_policy, selector_optimizer, 
                    selector_batch_log_probs, selector_batch_returns, 
                    selector_baseline
                )
                selector_batch_log_probs = []
                selector_batch_returns = []
        
        # Record History
        history['iterations'].append(iteration)
        history['cells_discovered'].append(len(archive))
        history['max_reward'].append(max(c['reward'] for c in archive.values()))
        
        if iteration % 100 == 0:
             pass # Silence intermediate output for multi-seed runs
            
    return archive, history, dyna_model, sweeping, selector_policy

# ============================================================================
# 7. Phase 2 Enhancement
# ============================================================================

# Re-define ActorCriticNetwork for standalone usage
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_states=256, num_actions=4, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        self.num_states = num_states
        self.fc1 = nn.Linear(num_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.policy_head(x), self.value_head(x)
    
    def get_action(self, state_idx, deterministic=False):
        device = next(self.parameters()).device
        state_onehot = torch.zeros(1, self.num_states, device=device)
        state_onehot[0, state_idx] = 1.0
        logits, value = self.forward(state_onehot)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

def backward_algorithm_ppo_enhanced(env, policy, reference_trajectory, archive=None, 
                                   use_near_miss=True, **kwargs):
    """
    Enhanced Backward Algorithm that generates a curriculum of starting trajectories,
    including near-miss states from the archive.
    """
    curriculum_trajectories = []
    traj_len = len(reference_trajectory)
    
    # Standard backward curriculum (indices -> trajectories)
    # Start indices: 90%, 70%, 50%, 30%, 0%
    indices = [
        int(0.9 * traj_len),
        int(0.7 * traj_len),
        int(0.5 * traj_len),
        int(0.3 * traj_len),
        0
    ]
    
    # Add standard trajectory segments
    for idx in indices:
        curriculum_trajectories.append(reference_trajectory[:idx])
    
    if use_near_miss and archive:
        goal_pos = (11, 11) # From map
        near_misses = []
        
        # Get cells in reference trajectory to avoid duplicates
        ref_cells = set()
        # We would need to rollout to know for sure, but we can skip this strict check for now
        # or assume reference_trajectory is efficient.
        
        for cell, data in archive.items():
            row = cell // 16
            col = cell % 16
            dist = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
            
            # If close to goal (e.g. within 5 steps) 
            if 0 < dist < 5:
                # Check if reward is < 1.0 (didn't reach goal)
                if data['reward'] < 1.0:
                    near_misses.append(data['trajectory'])
        
        print(f"Found {len(near_misses)} near-miss trajectories to add to curriculum.")
        
        # Insert near-misses at the beginning (easiest tasks)
        curriculum_trajectories = near_misses + curriculum_trajectories
        
    return curriculum_trajectories

if __name__ == "__main__":
    # Multi-seed test run across multiple maps
    NUM_SEEDS = 5
    print(f"Running Enhanced Go-Explore Phase 1 on multiple maps ({NUM_SEEDS} seeds each)...")
    
    results_comparison = {}

    for map_name, map_layout in maps.items():
        print(f"\n{'='*30}")
        print(f"Testing Map: {map_name}")
        print(f"{'='*30}")
        
        results_comparison[map_name] = {'Original': [], 'Enhanced': []}
        
        # 1. Original Algorithm
        print(f"Running Original Algorithm...")
        for seed in range(NUM_SEEDS):
            random.seed(42 + seed); np.random.seed(42 + seed); torch.manual_seed(42 + seed)
            env_orig = gym.make('FrozenLake-v1', desc=map_layout, is_slippery=False, render_mode=None)
            archive_orig, history_orig = go_explore_phase1(env_orig, max_iterations=500, k_explore=10, target_reward=1.0)
            results_comparison[map_name]['Original'].append(history_orig)
            print(f"  Seed {seed}: Solved {history_orig['solved_iteration']}, Discovered {len(archive_orig)}")
        
        # 2. Enhanced Algorithm
        print(f"Running Enhanced Algorithm...")
        for seed in range(NUM_SEEDS):
            random.seed(42 + seed); np.random.seed(42 + seed); torch.manual_seed(42 + seed)
            env_enh = gym.make('FrozenLake-v1', desc=map_layout, is_slippery=False, render_mode=None)
            archive_enh, history_enh, _, _, _ = go_explore_phase1_enhanced(
                env_enh, max_iterations=500, k_explore=10, target_reward=1.0,
                use_dyna=True, use_sweeping=True, use_learned_selector=True
            )
            results_comparison[map_name]['Enhanced'].append(history_enh)
            print(f"  Seed {seed}: Solved {history_enh['solved_iteration']}, Discovered {len(archive_enh)}")
    
    # Summary
    print("\n" + "="*80)
    print(f"{'Map':<15} | {'Algorithm':<10} | {'Success %':<10} | {'Mean Solved Iter':<18} | {'Mean Final Cells':<18}")
    print("-" * 80)
    
    for map_name, res in results_comparison.items():
        for algo in ['Original', 'Enhanced']:
            histories = res[algo]
            solved_counts = sum(1 for h in histories if h['solved_iteration'] is not None)
            success_rate = (solved_counts / NUM_SEEDS) * 100
            
            solved_iters = [h['solved_iteration'] for h in histories if h['solved_iteration'] is not None]
            mean_solved = np.mean(solved_iters) if solved_iters else float('inf')
            
            final_cells = [h['cells_discovered'][-1] for h in histories]
            mean_cells = np.mean(final_cells)
            
            mean_solved_str = f"{mean_solved:.1f}" if mean_solved != float('inf') else "N/A"
            
            print(f"{map_name:<15} | {algo:<10} | {success_rate:<10.1f} | {mean_solved_str:<18} | {mean_cells:<18.1f}")
        print("-" * 80)

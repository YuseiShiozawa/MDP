import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the grid size
grid_size = (20, 20)

# Define water puddle areas
pond1 = [(x, y) for x in range(0, 7) for y in range(6, 14)]
pond2 = [(x, y) for x in range(7, 17) for y in range(7, 17)]

# Define the states
states = [(x, y, theta) for x in range(grid_size[0]) for y in range(grid_size[1]) for theta in range(4)]

# Define actions including diagonal moves
actions = ['left', 'right', 'forward', 'left_forward', 'right_forward']

# Define rewards (as a dictionary where keys are states and values are rewards)
rewards = {state: -0.1 for state in states}
for x, y in pond1:
    for theta in range(4):
        rewards[(x, y, theta)] = -1 * 0.2  # Water 
for x, y in pond2:
    for theta in range(4):
        rewards[(x, y, theta)] = -1 * 0.2  # Water

# Define a goal state
goal_state = (0, 19, 0)
for theta in range(4):
    rewards[(0, 19, theta)] = 1  # Goal state

# Define function to get new state considering theta for diagonal moves
def get_new_state(state, action):
    x, y, theta = state
    if action == 'left':
        return (x, y, (theta - 1) % 4)
    elif action == 'right':
        return (x, y, (theta + 1) % 4)
    elif action == 'forward':
        if theta == 0:
            return (x, y + 1, theta) if y < grid_size[1] - 1 else state
        elif theta == 1:
            return (x + 1, y, theta) if x < grid_size[0] - 1 else state
        elif theta == 2:
            return (x, y - 1, theta) if y > 0 else state
        elif theta == 3:
            return (x - 1, y, theta) if x > 0 else state
    elif action == 'left_forward':
        if theta == 0:
            return (x - 1, y + 1, theta) if y < grid_size[1] - 1 and x > 0 else state
        elif theta == 1:
            return (x + 1, y + 1, theta) if y < grid_size[1] - 1 and x < grid_size[0] - 1 else state
        elif theta == 2:
            return (x + 1, y - 1, theta) if y > 0 and x < grid_size[0] - 1 else state
        elif theta == 3:
            return (x - 1, y - 1, theta) if y > 0 and x > 0 else state
    elif action == 'right_forward':
        if theta == 0:
            return (x + 1, y + 1, theta) if y < grid_size[1] - 1 and x < grid_size[0] - 1 else state
        elif theta == 1:
            return (x + 1, y - 1, theta) if y > 0 and x < grid_size[0] - 1 else state
        elif theta == 2:
            return (x - 1, y - 1, theta) if y > 0 and x > 0 else state
        elif theta == 3:
            return (x - 1, y + 1, theta) if y < grid_size[1] - 1 and x > 0 else state

# Define transition probabilities
transition_probabilities = {}
for state in states:
    transition_probabilities[state] = {}
    for action in actions:
        new_state = get_new_state(state, action)
        transition_probabilities[state][action] = {new_state: 1.0}




# Define value iteration algorithm
def value_iteration(states, actions, rewards, transition_probabilities, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}
    while True:
        delta = 0
        for state in states:
            v = V[state]
            V[state] = max(
                sum(
                    prob * (rewards.get(next_state, -1) + gamma * V.get(next_state, 0))
                    for next_state, prob in transition_probabilities[state][action].items()
                )
                for action in actions
            )
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        for action in actions:
            value = sum(
                prob * (rewards.get(next_state, -1) + gamma * V.get(next_state, 0))
                for next_state, prob in transition_probabilities[state][action].items()
            )
            if value > best_value:
                best_value = value
                best_action = action
        policy[state] = best_action
    return policy, V

# Get the optimal policy and value function
policy, value_function = value_iteration(states, actions, rewards, transition_probabilities)

# Simulate the policy
def simulate_policy(policy, start_state, steps=100):
    state = start_state
    path = [state]
    for _ in range(steps):
        action = policy[state]
        state = get_new_state(state, action)
        path.append(state)
        if state == goal_state:
            break
    return path

# Function to create a triangular patch for the robot
def create_triangle(state):
    x, y, theta = state
    if theta == 0:
        vertices = [(x - 0.3, y - 0.3), (x + 0.3, y - 0.3), (x, y + 0.3)]
    elif theta == 1:
        vertices = [(x - 0.3, y - 0.3), (x - 0.3, y + 0.3), (x + 0.3, y)]
    elif theta == 2:
        vertices = [(x - 0.3, y + 0.3), (x + 0.3, y + 0.3), (x, y - 0.3)]
    elif theta == 3:
        vertices = [(x + 0.3, y - 0.3), (x + 0.3, y + 0.3), (x - 0.3, y)]
    elif theta == 4:  # 斜め左上
        vertices = [(x - 0.3, y - 0.3), (x - 0.3, y + 0.3), (x + 0.3, y + 0.3)]
    elif theta == 5:  # 斜め右上
        vertices = [(x - 0.3, y + 0.3), (x + 0.3, y + 0.3), (x + 0.3, y - 0.3)]
    elif theta == 6:  # 斜め右下
        vertices = [(x + 0.3, y + 0.3), (x + 0.3, y - 0.3), (x - 0.3, y - 0.3)]
    elif theta == 7:  # 斜め左下
        vertices = [(x + 0.3, y - 0.3), (x - 0.3, y - 0.3), (x - 0.3, y + 0.3)]
    return plt.Polygon(vertices, color='red')

# Visualize the policy
fig, ax = plt.subplots()
ax.set_xlim(-0.5, grid_size[0] - 0.5)
ax.set_ylim(-0.5, grid_size[1] - 0.5)

# Mark water puddle areas
for x, y in pond1:
    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='blue', alpha=0.3))
for x, y in pond2:
    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='blue', alpha=0.3))

# Add rewards to the grid
for (x, y, theta), reward in rewards.items():
    ax.text(x, y, f'{reward:.1f}', color='black', fontsize=8, ha='center', va='center')

# Run the simulation
start_state = (0, 0, 0)  # Start from the top-right corner
path = simulate_policy(policy, start_state)

# Animation initialization
agent = create_triangle(start_state)
ax.add_patch(agent)
path_line, = ax.plot([], [], 'r-')
reward_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right')
cumulative_reward_text = ax.text(0.95, 0.10, '', transform=ax.transAxes, ha='right')

cumulative_reward = 0
goal_reached = False

def init():
    global cumulative_reward, goal_reached
    agent.set_xy(create_triangle(start_state).get_xy())
    path_line.set_data([], [])
    reward_text.set_text('')
    cumulative_reward_text.set_text('')
    cumulative_reward = 0
    goal_reached = False
    return agent, path_line, reward_text, cumulative_reward_text

def update(frame):
    global cumulative_reward, goal_reached
    x, y, theta = path[frame]
    agent.set_xy(create_triangle((x, y, theta)).get_xy())
    if frame > 0:
        x_prev, y_prev, _ = path[frame - 1]
        path_line.set_data([x_prev, x], [y_prev, y])
    reward = rewards[(x, y, theta)]
    if not goal_reached:
        cumulative_reward += reward
    reward_text.set_text(f'Reward: {reward:.1f}')
    cumulative_reward_text.set_text(f'Cumulative Reward: {cumulative_reward:.1f}')
    if (x, y, theta) == goal_state:
        goal_reached = True
    return agent, path_line, reward_text, cumulative_reward_text

# Animation generation
ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, repeat=False, interval=100)
plt.gca().invert_yaxis()

# Save animation as GIF

# Display the animation
plt.show()
ani.save('robot_simulation2.gif', writer='pillow', fps=10)

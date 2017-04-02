from Reinforcement_Learning import Agent
from Reinforcement_Learning import GridWorld as gw

if __name__ == "__main__":
    world = gw(5, 4)
    # detective = Agent()
    gamma = 0.5
    alpha = 0.5
    p = 0.5

    # Performing value iteration to find the optimal policy
    for _ in range(70):
        for row in world.world_grid:
            for square in row:
                square.value_iteration(gamma)
    world.display_world("world")
    world.display_world("reward")
    world.display_world("value")

    gamma = 0.9

    # Performing Q learning to find optimal policy by Policy iteration
    for _ in range(1, 5000):
        for row in world.world_grid:
            for square in row:
                square.learn_q_values(alpha, gamma, p)
        if _ % 10 == 0:
            p = 1 - (1 - p) / (2 - p)
            # print(p)
        # world.display_world("q_value")
        # print()
    world.display_world("q_value")
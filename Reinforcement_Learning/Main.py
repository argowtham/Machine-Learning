from Reinforcement_Learning import Agent
from Reinforcement_Learning import GridWorld as gw

if __name__ == "__main__":
    world = gw(5, 4)
    # detective = Agent()

    # Performing value iteration to find the optimal policy
    for _ in range(70):
        for row in world.world_grid:
            for square in row:
                square.value_iteration(0.5)
    world.display_world("world")
    world.display_world("reward")
    world.display_world("value")





import random
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map

if __name__ == '__main__':
    env = MetaDriveEnv(config=dict(num_scenarios=100, map=6))

    fig, axs = plt.subplots(4, 5, figsize=(10, 10), dpi=100)
    count = 0
    #We are going to draw 20 PG maps with 6 blocks each !)
    for i in range(4):
        for j in range(5):
            seed = random.randint(0, 1000)
            print(seed)
            env.seed(seed)
            count += 1
            env.reset()
            m = draw_top_down_map(env.current_map)
            ax = axs[i][j]
            ax.imshow(m, cmap="bone")
            ax.set_xticks([])
            ax.set_yticks([])
            print("Drawing {}-th map!".format(count))
    fig.suptitle("Top-down view of generated maps")
    plt.show()
    env.close()

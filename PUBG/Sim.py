import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import compress
try:
    import queue
except ImportError:
    import Queue as queue

from Actor import PlayerActor

ACTOR_NUM = 100
ADVANTAGE_NUM = 100

SQ_FIELD_WIDTH = 10
R_RESTRICT = 5

KILLING_DISTANCE = 0.2

killing_list = queue.deque([], 4)

actor_list = [PlayerActor(SQ_FIELD_WIDTH, ADVANTAGE_NUM, i) for i in range(ACTOR_NUM)]
advantage_points = np.random.uniform(-SQ_FIELD_WIDTH/2, SQ_FIELD_WIDTH/2, size=(ADVANTAGE_NUM, 2))
advantage_points[-1] = [0,0]
advantage_points_met = [False for _ in range(ADVANTAGE_NUM)]
counter = 0
survive_num = []

fig, ax = plt.subplots(1,2,figsize=(20,8))

TIMES_PER_STEP = 50
DROPPING_TIME = 10

def Area(t):
    step = t // TIMES_PER_STEP
    current_area = (0.8) ** step * 25 * np.pi
    remainder = t % TIMES_PER_STEP
    if remainder > (TIMES_PER_STEP - DROPPING_TIME):
        next_area = (0.8) ** (step + 1) * 25 * np.pi
        return current_area + (next_area - current_area) * (remainder - (TIMES_PER_STEP - DROPPING_TIME)) / DROPPING_TIME

    return current_area

def massecre():
    global actor_list, killing_list
    positions = np.array(list(map(lambda x: x.pos, actor_list)))
    index_random = np.random.permutation(len(actor_list))
    alive = [True for _ in range(len(actor_list))]
    for i in index_random:
        if not alive[i]:
            continue
        for j in index_random:
            if not alive[j] or i == j:
                continue
            if np.linalg.norm(positions[i] - positions[j]) < KILLING_DISTANCE:
                sum_power = actor_list[i].power + actor_list[j].power
                to_live, to_die = np.random.choice([i,j], size=2, replace=False, \
                    p=[float(actor_list[i].power)/sum_power, float(actor_list[j].power)/sum_power])
                actor_list[to_live].power = max(actor_list[to_live].power - 30, 50)
                alive[to_die] = False
                killing_list.append((actor_list[to_live].player_name,actor_list[to_die].player_name))
                actor_list[to_live].kill += 1

    actor_list = list(compress(actor_list, alive))




def render(frame):
    global R_RESTRICT, counter, actor_list, survive_num, advantage_points_met

    R_RESTRICT = np.sqrt(Area(counter) / np.pi)

    ax[0].cla()
    ax[1].cla()
    if len(actor_list) == 1:
        ax[1].text(150, 60, 'Player {} killed {} people'.format(actor_list[0].player_name, actor_list[0].kill))

    for i in np.random.permutation(len(actor_list)):
        actor_list[i].move(R_RESTRICT, advantage_points)
        actor_list[i].check_advantage_point(advantage_points, advantage_points_met)
    counter += 1

    survive_num.append(len(actor_list))
    ax[1].plot(np.arange(counter), survive_num)
    ax[0].set_xlim(-SQ_FIELD_WIDTH/2, SQ_FIELD_WIDTH/2)
    ax[0].set_ylim(-SQ_FIELD_WIDTH/2, SQ_FIELD_WIDTH/2)
    ax[1].set_xlim(0,300)
    ax[1].set_ylim(0,100)

    ax[0].text(4.2,4.2,'Survival {}'.format(len(actor_list)))

    massecre()

    # plot advantage
    ax[0].scatter(advantage_points[:,0],advantage_points[:,1], c='r')

    # plot player
    positions = np.array(list(map(lambda x: x.pos, actor_list)))
    ax[0].scatter(positions[:,0],positions[:,1],c='b')
    for player, x, y in zip(actor_list, positions[:,0], positions[:,1]):
        ax[0].annotate(str(player.player_name), (x,y))


    # plot circle
    circle = plt.Circle((0,0), R_RESTRICT, color='g', fill=False)
    ax[0].add_artist(circle)

    # Showing Casualties list
    texts = list(map(lambda x: '{} killed {}'.format(x[0],x[1]), killing_list))
    ax[1].text(220,80, ('\n').join(texts))

    #if (counter+1) % 50 == 0:
    #    R_RESTRICT *= 0.8


if __name__ == "__main__":
    ani = FuncAnimation(fig, render, frames=23, interval=300)
    ani.save('anim.gif',writer='imagemagick')
    # plt.show()

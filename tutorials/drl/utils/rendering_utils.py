import math
import matplotlib.colors as colours
from mdp import *
import matplotlib.pyplot as plt

COLOURS = {
    'red': [200, 50, 50],
    'green': [90, 165, 90],
    'blue': [0, 0, 255],
    'purple': [112, 39, 195],
    'grey': [150, 150, 150],
    'white': [255, 255, 255],
    'black': [0, 0, 0],
    'yellow': [255, 255, 0]
}

# Action symbols from gridworld
LEFT = '\u25C4'
UP = '\u25B2'
RIGHT = '\u25BA'
DOWN = '\u25BC'

'''Draw the grid lines to differentiate discrete states'''
def draw_grid_lines(i, j, img):
    img[i][j] = COLOURS['black']


'''Draw a triangle based on size, center, direction and colour'''
def draw_triangle(tile_origin, tile_size, img, colour='red', direction='up'):
    origin_x, origin_y = tile_origin
    for x in range(origin_x + 1, origin_x + tile_size - 1):
        for y in range(origin_y + 1, origin_y + tile_size -1):
            if direction == DOWN:
                if y < origin_y + tile_size//2 and x + y < origin_x + origin_y + tile_size and x - origin_x > y - origin_y:
                    img[y][x] = colour
            elif direction == UP:
                if y > origin_y + tile_size // 2 and x + y > origin_x + origin_y + tile_size and x - origin_x < y - origin_y:
                    img[y][x] = colour
            elif direction == LEFT:
                if x < origin_x + tile_size // 2 and x - origin_x < y - origin_y and x + y < origin_x + origin_y + tile_size:
                    img[y][x] = colour
            elif direction == RIGHT:
                if x > origin_x + tile_size // 2 and x - origin_x > y - origin_y and x + y > origin_x + origin_y + tile_size:
                    img[y][x] = colour
            else:
                raise ValueError("Invalid direction")


''' Render each Q value which forms a triangle in the grid representation of the Q-function.'''
def render_action_q_value(tileSize, x, y, action, q_values, img, show_text=False, text_size=12, h_text_offset=0, v_text_offset=0, rewardMax=1, rewardMin=1):
    value = q_values.get_q_value((x, y), action) #MDP.get_q_value(q_values, (x, y), action=action)
    colour = COLOURS['red'] if value < 0 else COLOURS['green']  # make colour red if value is negative, otherwise make it green
    scaling_factor = rewardMin if value < 0 else rewardMax
    colour = list(map(lambda c: int(c * math.fabs(value/scaling_factor)),
                      colour))  # scale the colour by the reward (make extremes more vivid)
    draw_triangle((x * tileSize, y * tileSize), tileSize, img, colour=colour, direction=action)
    if show_text:
        plt.text(x=x * tileSize + tileSize // 2 + h_text_offset, y=y * tileSize + tileSize // 2 + v_text_offset,
                 s=f'{value:.2f}', size=text_size, verticalalignment='center', horizontalalignment='center', color='white')


''' Render each Q value which forms a triangle in the grid representation of the Q-function.'''
def render_action_probability(tileSize, x, y, action, prob, text_size=6, h_text_offset=0, v_text_offset=0):
    plt.text(x=x * tileSize + tileSize // 2 + h_text_offset, y=y * tileSize + tileSize // 2 + v_text_offset,
             s=f'{prob:.2f}\n{action}', size=text_size, verticalalignment='center', horizontalalignment='center', color='white')


'''render blocked tile as a black and white criss-cross'''
def render_blocked_tile(i, j, img):
    img[i][j] = COLOURS['grey']
    """
    EDIT
    if i % 2 == 0 or j % 2 == 0:
        img[i][j] = COLOURS['black']
    else:
        img[i][j] = COLOURS['white']
    """

def render_full_blocked_tile(x, y, tile_size, img):
    for i in range(x, x+tile_size):
        for j in range(y, y+tile_size):
            if i % 2 == 0 or j % 2 == 0:
                img[j][i] = COLOURS['black']
            else:
                img[j][i] = COLOURS['white']


def render_full_goal_tile(x, y, tile_size, img, reward, rewardMax, rewardMin):
    for i in range(x, x+tile_size):
        for j in range(y, y+tile_size):
            if reward > 0:
                img[j][i] = [0, int(255 * reward / rewardMax), 0]
            else:
                img[j][i] = [int(255 * reward / rewardMin), 0, 0]


'''render the agent as a circle'''
def render_agent(i, j, img, center_x, center_y, radius):
    h_dist = math.fabs(center_x - j)
    v_dist = math.fabs(center_y - i)
    if h_dist ** 2 + v_dist ** 2 <= radius ** 2:
        img[i][j] = COLOURS['yellow']
    else:
        img[i][j] = COLOURS['white']


'''
Render the goal as a coloured cell. THe color depend on the value of the goal.
Positive values are green, with brighter green representing higher reward.
Negative values are red, with brighter red representing lower reward. 
'''
def render_goal(i, j, img, reward, reward_max=1, reward_min=-1):
    """
    EDIT
    if reward > 0:
        img[i][j] = [0, int(255 * reward / reward_max), 0]
    else:
        img[i][j] = [int(255 * reward / reward_min), 0, 0]
    """
    img[i][j] = COLOURS['white']
    tileSize = 32
    plt.text(x = i* tileSize/2 , y=j* tileSize/2 , s=f'{reward:.2f}', size=12)
    #plt.text(x=i * tileSize + tileSize // 2 + 8, y=j * tileSize + tileSize // 2 + 8,
    #             s=f'{reward:.2f}', size=12, verticalalignment='center', horizontalalignment='center', color='white')

'''
Matplotlib doesn't have an inbuilt red to green colour map with white in the middle.
So we can just make our own.
'''
def make_red_white_green_cmap():
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.0, 0.0))
             }

    # Create the colormap using the dictionary
    return colours.LinearSegmentedColormap('GnRd', cdict)

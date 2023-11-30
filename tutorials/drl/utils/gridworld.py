from collections import defaultdict

from mdp import *
from rendering_utils import *


class GridWorld(MDP):
    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')
    LEFT = '\u25C4'
    UP = '\u25B2'
    RIGHT = '\u25BA'
    DOWN = '\u25BC'

    def __init__(
        self,
        noise=0.1,
        width=4,
        height=3,
        discount_factor=0.9,
        blocked_states=[(1, 1)],
        action_cost=0.0,
        initial_state=(0, 0),
        goals=None,
    ):
        self.noise = noise
        self.width = width
        self.height = height
        self.blocked_states = blocked_states
        self.discount_factor = discount_factor
        self.action_cost = action_cost
        self.initial_state = initial_state
        if goals is None:
            self.goal_states = dict(
                [((width - 1, height - 1), 1), ((width - 1, height - 2), -1)]
            )
        else:
            self.goal_states = dict(goals)

        # A list of lists thatrecords all rewards given at each step
        # for each episode of a simulated gridworld
        self.rewards = []
        # The rewards for the current episode
        self.episode_rewards = []


    def get_states(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blocked_states:
                    states.append((x, y))
        return states

    def get_actions(self, state=None):

        actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.TERMINATE]
        if state is None:
            return actions

        valid_actions = []
        for action in actions:
            for (new_state, probability) in self.get_transitions(state, action):
                if probability > 0:
                    valid_actions.append(action)
                    break
        return valid_actions

    def get_initial_state(self):
        self.episode_rewards = []
        return self.initial_state

    def get_goal_states(self):
        return self.goal_states

    def valid_add(self, state, new_state, probability):
        # If the next state is blocked, stay in the same state
        if probability == 0.0:
            return []

        if new_state in self.blocked_states:
            return [(state, probability)]

        # Move to the next space if it is not off the grid
        (x, y) = new_state
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            return [((x, y), probability)]

        # If off the grid, state in the same state
        return [(state, probability)]

    def get_transitions(self, state, action):
        transitions = []

        if state == self.TERMINAL:
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return []

        # Probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if state in self.get_goal_states().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == self.UP:
            transitions += self.valid_add(state, (x, y + 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.DOWN:
            transitions += self.valid_add(state, (x, y - 1), straight)
            transitions += self.valid_add(state, (x - 1, y), self.noise)
            transitions += self.valid_add(state, (x + 1, y), self.noise)

        elif action == self.RIGHT:
            transitions += self.valid_add(state, (x + 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        elif action == self.LEFT:
            transitions += self.valid_add(state, (x - 1, y), straight)
            transitions += self.valid_add(state, (x, y - 1), self.noise)
            transitions += self.valid_add(state, (x, y + 1), self.noise)

        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

        return transitions

    def get_reward(self, state, action, new_state):
        reward = 0.0
        if state in self.get_goal_states().keys() and new_state == self.TERMINAL:
            reward = self.get_goal_states().get(state)
        else:
            # default：0
            reward = self.action_cost
        step = len(self.episode_rewards)
        self.episode_rewards += [reward * (self.discount_factor ** step)]
        return reward

    def get_discount_factor(self):
        return self.discount_factor

    def is_terminal(self, state):
        if state == self.TERMINAL:
            #self.rewards += [self.episode_rewards]
            return True
        return False

    """
        Returns a list of lists, which records all rewards given at each step
        for each episodeof a simulated gridworld
    """

    def get_rewards(self):
        return self.rewards

    """
        Create a gridworld from an array of strings: one for each line
        - First line is rewards as a dictionary from cell to value: {'A': 1, ...}
        - space is an empty cell
        - # is a blocked cell
        - @ is the agent (initial state)
        - new 'line' is a new row
        - a letter is a cell with a reward for transitioning
          into that cell. The reward defined by the first line.
    """

    @staticmethod
    def create(string):
        # Parse the reward on the first line
        import ast

        rewards = ast.literal_eval(string[0])

        width = 0
        height = len(string) - 1

        blocked_cells = []
        initial_state = (0, 0)
        goals = []
        row = 0
        for next_row in string[1:]:
            column = 0
            for cell in next_row:
                if cell == "#":
                    blocked_cells += [(column, row)]
                elif cell == "@":
                    initial_state = (column, row)
                elif cell.isalpha():
                    goals += [((column, row), rewards[cell])]
                column += 1
            width = max(width, column)
            row += 1
        return GridWorld(
            width=width,
            height=height,
            blocked_states=blocked_cells,
            initial_state=initial_state,
            goals=goals,
        )

    @staticmethod
    def open(file):
        file = open(file, "r")
        string = file.read().splitlines()
        file.close()
        return GridWorld.create(string)

    @staticmethod
    def matplotlib_installed():
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            return True
        except ModuleNotFoundError:
            return False

    """ Visualise a Grid World problem """

    def visualise(self, agent_position=None, title="", grid_size=1.5, gif=False):
        if self.matplotlib_installed():
            return self.visualise_as_image(agent_position=agent_position, title=title, grid_size=grid_size, gif=gif)
        else:
            print(self.to_string(title=title))

    """ Visualise a Grid World value function """
    def visualise_value_function(self, value_function, title="", grid_size=1.5, gif=False):
        if self.matplotlib_installed():
            return self.visualise_value_function_as_image(value_function, title=title, grid_size=grid_size, gif=gif)
        else:
            print(self.value_function_to_string(value_function, title=title))

    def visualise_q_function(self, qfunction, title="", grid_size=2.0, gif=False):
        if self.matplotlib_installed():
            return self.visualise_q_function_as_image(qfunction, title=title, grid_size=grid_size, gif=gif)
        else:
            print(self.q_function_to_string(qfunction, title=title))

    def visualise_policy(self, policy, title="", grid_size=1.5, gif=False):
        if self.matplotlib_installed():
            return self.visualise_policy_as_image(policy, title=title, grid_size=grid_size, gif=gif)
        else:
            print(self.policy_to_string(policy, title=title))

    def visualise_stochastic_policy(self, policy, title="", grid_size=1.5, gif=False):
        if self.matplotlib_installed():
            return self.visualise_stochastic_policy_as_image(policy, title=title, grid_size=grid_size, gif=gif)
        else:
            # TODO make a stochastic policy to string
            pass

    """ Visualise a grid world problem as a formatted string """
    def to_string(self, title=""):
        left_arrow = "\u25C4"
        up_arrow = "\u25B2"
        right_arrow = "\u25BA"
        down_arrow = "\u25BC"


        space = " |              "
        block = " | #############"

        line = "  "
        for x in range(self.width):
            line += "--------------- "
        line += "\n"

        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.get_goal_states().keys():
                    result += space
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += " |       {}      ".format(up_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |     _____    "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |    ||o  o|   "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " | {}  ||  * |  {}".format(left_arrow, right_arrow)
                elif (x, y) in self.blocked_states:
                    result += block
                elif (x, y) in self.get_goal_states().keys():
                    result += " |     {:+0.2f}    ".format(
                        self.get_goal_states()[(x, y)]
                    )
                else:
                    result += " | {}           {}".format(left_arrow, right_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |    ||====|   ".format(left_arrow, right_arrow)
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.get_initial_state():
                    result += " |     -----    "
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.get_goal_states().keys():
                    result += space
                elif (x, y) in self.blocked_states:
                    result += block
                else:
                    result += " |       {}      ".format(down_arrow)
            result += " |\n"
            result += line
        return result

    """ Convert a grid world value function to a formatted string """

    def value_function_to_string(self, values, title=""):
        line = " {:-^{n}}\n".format("", n=len(" | +0.00") * self.width + 1)
        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " | #####"
                else:
                    result += " | {:+0.2f}".format(values.get_value((x, y)))
            result += " |\n"
            result += line

        return result

    """ Convert a grid world Q function to a formatted string """

    def q_function_to_string(self, qfunction, title=""):
        left_arrow = "\u25C4"
        up_arrow = "\u25B2"
        right_arrow = "\u25BA"
        down_arrow = "\u25BC"

        space = " |               "

        line = "  "
        for x in range(self.width):
            line += "---------------- "
        line += "\n"

        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |       {}       ".format(up_arrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(
                        qfunction.get_q_value((x, y), self.UP)
                    )
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " |     #####     "
                elif (x, y) in self.get_goal_states().keys():
                    result += " |     {:+0.2f}     ".format(
                        self.get_goal_states()[(x, y)]
                    )
                else:
                    result += " | {}{:+0.2f}  {:+0.2f}{}".format(
                        left_arrow,
                        qfunction.get_q_value((x, y), self.LEFT),
                        qfunction.get_q_value((x, y), self.RIGHT),
                        right_arrow,
                    )
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(
                        qfunction.get_q_value((x, y), self.DOWN)
                    )
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blocked_states or (
                    x,
                    y,
                ) in self.get_goal_states().keys():
                    result += space
                else:
                    result += " |       {}       ".format(down_arrow)
            result += " |\n"
            result += line
        return result

    """ Convert a grid world policy to a formatted string """

    def policy_to_string(self, policy, title=""):
        line = " {:-^{n}}\n".format("", n=len(" |  N ") * self.width + 1)
        result = " " + title + "\n"
        result += line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    result += " | ###"
                elif policy.select_action((x, y)) == self.TERMINATE:
                    result += " | {:+0d} ".format(self.goal_states[(x, y)])
                else:
                    result += " |  " + policy.select_action((x, y)) + " "
            result += " |\n"
            result += line

        return result


    """ Initialise a gridworld grid """
    def initialise_grid(self, grid_size=1.5):
        fig = plt.figure(figsize=(self.width * grid_size, self.height * grid_size))
        plt.subplots_adjust(top=0.92, bottom=0.01, right=1, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1)

        # Initialise the map to all white
        img = [[COLOURS['white'] for _ in range(self.width)] for _ in range(self.height)]

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                if (x, y) in self.goal_states:
                    img[y][x] = COLOURS['red'] if self.goal_states[(x, y)] < 0 else COLOURS['green']
                elif (x, y) in self.blocked_states:
                    img[y][x] = COLOURS['grey']

        ax.xaxis.set_ticklabels([])  # clear x tick labels
        ax.axes.yaxis.set_ticklabels([])  # clear y tick labels
        ax.tick_params(which='both', top=False, left=False, right=False, bottom=False)
        ax.set_xticks([w - 0.5 for w in range(0, self.width, 1)])
        ax.set_yticks([h - 0.5 for h in range(0, self.height, 1)])
        ax.grid(color='lightgrey')
        return fig, ax, img

    """ visualise the gridworld problem as a matplotlib image """

    def visualise_as_image(self, agent_position=None, title="", grid_size=1.5, gif=False):
        fig, ax, img = self.initialise_grid(grid_size=grid_size)
        current_position = (
            self.get_initial_state() if agent_position is None else agent_position
        )

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                if (x, y) == current_position:
                    ax.scatter(x, y, s=2000, marker='o', edgecolors='none')
                elif (x, y) in self.goal_states:
                    plt.text(
                        x,
                        y,
                        f"{self.get_goal_states()[(x, y)]:+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
        im = plt.imshow(img, origin="lower")
        plt.title(title)
        if gif:
            return fig, ax, im
        else:
            return fig

    """Render each tile individually depending on the current state of the cell"""

    def render_tile(self, x, y, tile_size, img, tile_type=None):
        ymin = y * tile_size
        ymax = (y + 1) * tile_size
        xmin = x * tile_size
        xmax = (x + 1) * tile_size

        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if i == ymin or i == ymax - 1 or j == xmin or j == xmax + 1:
                    draw_grid_lines(i, j, img)
                else:
                    if tile_type == "goal":
                        render_goal(
                            i,
                            j,
                            img,
                            reward=self.goal_states[(x, y)],
                            reward_max=max(self.get_goal_states().values()),
                            reward_min=min(self.get_goal_states().values()),
                        )
                    elif tile_type == "blocked":
                        render_blocked_tile(i, j, img)
                    elif tile_type == "agent":
                        render_agent(
                            i,
                            j,
                            img,
                            center_x=xmin + tile_size / 2,
                            center_y=ymin + tile_size / 2,
                            radius=tile_size / 4,
                        )
                    elif tile_type == "empty":
                        img[i][j] = [255, 255, 255]
                    else:
                        raise ValueError("Invalid tile type")

    """ Visualise the value function """

    def visualise_value_function_as_image(self, V, title="", grid_size=1.5, gif=False):
        if not gif:
            fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                value = V[(x, y)]
                if (x, y) not in self.blocked_states:
                    text = plt.text(
                        x,
                        y,
                        f"{float(value):+0.2f}",
                        fontsize="x-large",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color='lightgrey' if value == 0.0 else 'black',
                    )
                    texts.append(text)
        if gif:
            return texts
        else:
            ax.imshow(img, origin="lower")
            plt.title(title)
#             plt.show()

    """ Visualise the value function using a heat-map where green is high value and
    red is low value
    """

    def visualise_value_function_as_heatmap(self, value_function, title=""):
        values = [[0 for _ in range(self.width)] for _ in range(self.height)]
        fig, ax = self.initialise_grid()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.blocked_states:
                    plt.text(
                        x,
                        y,
                        "#",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                else:
                    values[y][x] = value_function.get_value((x, y))
                    plt.text(
                        x,
                        y,
                        f"{values[y][x]:.2f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
        plt.imshow(values, origin="lower", cmap=make_red_white_green_cmap())
        plt.title(title)
        plt.show()

    """ Visualise the Q-function with matplotlib """

    def visualise_q_function_as_image(self, qfunction, title="", grid_size=2.0, gif=False):
        if not gif:
            fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.goal_states:
                    # gif player handles goal state rendering
                    if not gif:
                        texts.append(plt.text(
                            x,
                            y,
                            f"{self.get_goal_states()[(x,y)]:+0.2f}",
                            fontsize="large",
                            horizontalalignment="center",
                            verticalalignment="center",
                        ))
                elif (x, y) not in self.blocked_states:
                    up_value = qfunction.get_q_value((x, y), self.UP)
                    down_value = qfunction.get_q_value((x, y), self.DOWN)
                    left_value = qfunction.get_q_value((x, y), self.LEFT)
                    right_value = qfunction.get_q_value((x, y), self.RIGHT)
                    texts.append(plt.text(
                        x,
                        y + 0.35,
                        f"{up_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="center",
                        verticalalignment="top",
                        color='lightgrey' if up_value == 0.0 else 'black',
                    ))
                    texts.append(plt.text(
                        x,
                        y - 0.35,
                        f"{down_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color='lightgrey' if down_value == 0.0 else 'black',
                    ))
                    texts.append(plt.text(
                        x - 0.45,
                        y,
                        f"{left_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="left",
                        verticalalignment="center",
                        color='lightgrey' if left_value == 0.0 else 'black'
                    ))
                    texts.append(plt.text(
                        x + 0.45,
                        y,
                        f"{right_value:+0.2f}",
                        fontsize="medium",
                        horizontalalignment="right",
                        verticalalignment="center",
                        color='lightgrey' if right_value == 0.0 else 'black'
                    ))
                    plt.plot([x-0.5, x+0.5], [y-0.5, y+0.5], ls='-', lw=1, color='lightgrey')
                    plt.plot([x + 0.5, x - 0.5], [y - 0.5, y + 0.5], ls='-', lw=1, color='lightgrey')
        if gif:
            return texts
        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()

    """ Visualise the Q-function with a matplotlib visual"""

    def visualise_q_function_rendered(self, q_values, title="", tile_size=32, show_text=False):
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = [[[0, 0, 0] for _ in range(width_px)] for _ in range(height_px)]

        # provide these to scale the colours between the highest and lowest value
        reward_max = max(self.get_goal_states().values())
        reward_min = min(self.get_goal_states().values())
        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                # Draw in the blocked states as a black and white mesh
                if (x, y) in self.blocked_states:
                    render_full_blocked_tile(
                        x * tile_size, y * tile_size, tile_size, img
                    )
                    continue
                # Draw goal states
                if (x, y) in self.goal_states:
                    render_full_goal_tile(
                        x * tile_size,
                        y * tile_size,
                        tile_size,
                        img,
                        reward=self.goal_states[(x, y)],
                        rewardMax=reward_max,
                        rewardMin=reward_min,
                    )
                    continue

                # Draw the action value for action available in each cell
                # Break the grid up into 4 sections, using triangles that meet
                # in the middle. The base of the triangle points toward the
                # direction of the action
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.UP,
                    q_values,
                    img,
                    show_text,
                    v_text_offset=8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.DOWN,
                    q_values,
                    img,
                    show_text,
                    v_text_offset=-8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.LEFT,
                    q_values,
                    img,
                    show_text,
                    h_text_offset=-8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )
                render_action_q_value(
                    tile_size,
                    x,
                    y,
                    self.RIGHT,
                    q_values,
                    img,
                    show_text,
                    h_text_offset=8,
                    rewardMax=reward_max,
                    rewardMin=reward_min,
                )

        ax.imshow(img, origin="lower", interpolation="bilinear")
        plt.title(title)
        plt.axis("off")
        plt.show()

    """ Visualise the policy of the agent with a matplotlib visual """

    def visualise_policy_as_image(self, policy, title="", grid_size=1.5, gif=False):
        # Map from basic unicode to prettier arrows
        arrow_map = {self.UP:'\u2191',
                     self.DOWN:'\u2193',
                     self.LEFT:'\u2190',
                     self.RIGHT:'\u2192',
                    }
        if not gif:
            fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.blocked_states and (x, y) not in self.goal_states:
                    if policy[(x, y)] != self.TERMINATE:
                        action = arrow_map[policy[(x, y)]]
                        fontsize = "xx-large"
                    texts.append(plt.text(
                                x,
                                y,
                                action,
                                fontsize=fontsize,
                                horizontalalignment="center",
                                verticalalignment="center",
                            ))
                elif (x, y) in self.goal_states:
                    # gif player handles goal state rendering
                    if not gif:
                        plt.text(
                            x,
                            y,
                            f"{self.get_goal_states()[(x, y)]:+0.2f}",
                            fontsize="x-large",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )
        if gif:
            return texts
        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()

    def execute(self, state, action):
        if state in self.goal_states:
            self.rewards += [self.episode_rewards]
            return MDP.execute(self, state=state, action=self.TERMINATE)
        return super().execute(state, action)

    def visualise_stochastic_policy_as_image(self, policy, title="", grid_size=1.5, gif=False):
        if not gif:
            fig, ax, img = self.initialise_grid(grid_size=grid_size)
        texts = []

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                prob_left = policy.get_probability((x, y), self.LEFT)
                prob_right = policy.get_probability((x, y), self.RIGHT)
                if self.height > 1:
                    prob_up = policy.get_probability((x, y), self.UP)
                    prob_down = policy.get_probability((x, y), self.DOWN)

                if (x, y) in self.goal_states:
                    # gif player handles goal state rendering
                    if not gif:
                        plt.text(
                            x,
                            y,
                            f"{self.get_goal_states()[(x, y)]:+0.2f}",
                            fontsize="x-large",
                            horizontalalignment="center",
                            verticalalignment="center",
                        )
                elif (x, y) not in self.blocked_states:
                    if self.height > 1:
                        texts.append(plt.text(
                            x,
                            y,
                            f"{prob_up:0.2f}\n{self.UP}\n{prob_left:0.2f}{self.LEFT} {self.RIGHT}{prob_right:0.2f}\n{self.DOWN}\n{prob_down:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        ))
                    else:
                        texts.append(plt.text(
                            x,
                            y,
                            f"{prob_left:0.2f}{self.LEFT} {self.RIGHT}{prob_right:0.2f}",
                            fontsize="medium",
                            horizontalalignment="center",
                            verticalalignment="center",
                        ))
        if gif:
            return texts
        ax.imshow(img, origin="lower")
        plt.title(title)
        plt.show()
        return fig


class CliffWorld(GridWorld):
    def __init__(
        self,
        noise=0.0,
        discount_factor=1.0,
        width=6,
        height=4,
        blocked_states=[],
        action_cost=-0.05,
        goals=[((1, 0), -5), ((2, 0), -5), ((3, 0), -5), ((4, 0), -5), ((5, 0), 0)],
    ):
        super().__init__(
            noise=noise,
            discount_factor=discount_factor,
            width=width,
            height=height,
            blocked_states=blocked_states,
            action_cost=action_cost,
            goals=goals,
        )


class OneDimensionalGridWorld(GridWorld):
    """ A one dimensional GridWorld class to use with the
    Logistic regression policy gradient.
    This allows actions [left, right] and terminates when the agent reaches the
    goal state without having to use a terminate action.
    """

    def __init__(
        self,
        noise=0.1,
        width=4,
        discount_factor=0.9,
        action_cost=0.0,
        initial_state=(0, 0),
        goals=[((0, 0), -1), ((10, 0), 1)],
    ):
        super().__init__(
            noise=noise,
            width=width,
            height=1,
            blocked_states=[],
            discount_factor=discount_factor,
            action_cost=action_cost,
            initial_state=initial_state,
            goals=goals,
        )

    def execute(self, state, action):
        # If we are in a goal state then terminate automatically execute
        # a terminate action to immediately terminate
        if state in self.goal_states:
            self.rewards += [self.episode_rewards]
            return MDP.execute(self, state=state, action=self.TERMINATE)
        return super().execute(state, action)


if __name__ == "__main__":
    small = GridWorld(width=8, height=6)
    small.visualise_as_image(title="Small")

    medium = gridworld = GridWorld(width=16, height=12)
    medium.visualise_as_image(title="Medium")

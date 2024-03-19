import collections
import numpy as np
import math

class GameState:
    def __init__(self, to_play=1):
        self.to_play = to_play
    def play(self, action):
        return GameState(to_play=-self.to_play)

class UCTNode():
    def __init__(self, state, action, parent=None):
        # self.id = id
        self.state = state
        self.action = action
        
        self.is_expanded = False
        
        # self.parent.child_total_value[self.action]
        # self.parent.child_number_visits[self.action]
        # 指向self
        self.parent = parent  # Optional[UCTNode]
        
        self.children = {}  # Dict[action, UCTNode]
        self.child_priors = np.zeros([10], dtype=np.float32)
        self.child_total_value = np.zeros([10], dtype=np.float32)
        self.child_number_visits = np.zeros([10], dtype=np.float32)

    # Ni
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value
        
    # ti
    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self) -> np.ndarray:
        return self.child_total_value / (1 + self.child_number_visits)

    # pUCT
    # https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture19/lecture19.pdf
    def child_U(self) -> np.ndarray:
        return math.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits))

    def best_child(self) -> int:
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_action = current.best_child()
            current = current.maybe_add_child(best_action)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, action):
        if action not in self.children:
            self.children[action] = UCTNode(
                self.state.play(action), action, parent=self)
        return self.children[action]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate * self.state.to_play)
            current = current.parent

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def print_tree_level_width(root: UCTNode):
    if not root:
        return
    
    queue = [(root, 0)]  # 初始化队列，元素为 (节点, 层级)
    current_level = 0
    level_nodes = []

    while queue:
        node, level = queue.pop(0)  # 从队列中取出当前节点和它的层级
        # 当进入新的一层时，打印上一层的信息并重置
        if level > current_level:
            print(f"Level {current_level} width: {len(level_nodes)}")
            level_nodes = [f'{node.action}']  # 重置当前层的节点列表
            current_level = level
        else:
            level_nodes.append(f'{node.action}')
        
        # 将当前节点的所有子节点加入队列
        for child in node.children.values():
            queue.append((child, level + 1))
    
    # 打印最后一层的信息
    print(f"Level {current_level} width: {len(level_nodes)}")
            
def UCT_search(state, num_reads):
    root = UCTNode(state, action=None, parent=DummyNode())
    for _ in range(num_reads):
        # 每次都是从根节点出发
        leaf = root.select_leaf()
        # child_priors: [0, 1]
        child_priors, value_estimate = NeuralNet().evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    print_tree_level_width(root)
    return root.best_child()


class NeuralNet():
    @classmethod
    def evaluate(self, game_state):
        # return policy_network(state), value_network(state)
        # policy_network(state): return pi(a|s)
        # value_network(state): return v(s)
        return np.random.random([10]), np.random.random()


num_reads = 100000
import time
tick = time.time()
UCT_search(GameState(), num_reads)
tock = time.time()
print("Took %s sec to run %s times" % (tock - tick, num_reads))
import resource
print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
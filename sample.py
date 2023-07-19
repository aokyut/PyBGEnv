from PyBGEnv import qubic
from random import choice
from tqdm import tqdm

env = qubic

def show(board):
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = 16 * i + 4 * j + k
                if board[idx] == 1:
                    print("O", end="")
                elif board[idx + 64] == 1:
                    print("X", end="")
                else:
                    print("-", end="")
            print("  |  ", end="")
        print("")
    print("")



def play(agent1, agent2):
    state = env.init()
    player = 0
    agents = [agent1, agent2]
    t = 0
    while True:
        t += 1
        action_mask = env.get_action_mask(state)
        agent = agents[player]
        action = agent.get_action(env, state, action_mask)
        # print(action)
        next_state = env.get_next(state, action, player)
        state = next_state
        # show(state)
        res = env.is_win(next_state, player)
        if t > 64: 
            exit(0)
        if res:
            if player == 0:
                return 1
            else:
                return -1
        if env.is_draw(next_state):
            return 0
        player = 1 - player

class RandomAgent:
    name = "Random"
    def get_action(self, env, state, action_mask):
        valid_action = env.valid_actions(state, env.current_player(state))
        return choice(valid_action)

class Minimax:
    def __init__(self, depth):
        self.name = f"MiniMax" + str(depth)
        self.depth = depth
    def get_action(self, env, state, action_mask):
        action = env.minimax_action(state, env.current_player(state), self.depth)
        return action

count = 0
count_draw = 0

agent1 = Minimax(1)
agent2 = Minimax(2)

for i in tqdm(range(100)):
    res = play(agent1, agent2)
    if res == 1:
        count += 1
    if res == 0:
        count_draw += 1
    res = play(agent2, agent1)
    if res == -1:
        count += 1
    if res == 0:
        count_draw += 1
print(count)
print(count_draw)
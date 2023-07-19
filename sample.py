from PyBGEnv import qubic
from random import choice

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
    t = 0
    while True:
        t += 1
        action_mask = env.get_action_mask(state)
        action = agent1.get_action(env, state, action_mask)
        next_state = env.get_next(state, action, player)
        state = next_state
        res = env.is_win(next_state, player)
        if res:
            if t < 10:
                assert False
            show(state)
            if player == 0:
                return 1
            else:
                return -1
        if env.is_draw(next_state):
            if t < 10:
                assert False
            show(state)
            return 0
        player = 1 - player

class RandomAgent:
    name = "Random"
    def get_action(self, env, state, action_mask):
        valid_action = env.valid_actions(state, env.current_player(state))
        return choice(valid_action)

for i in range(100):
    play(RandomAgent(), RandomAgent())
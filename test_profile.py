from line_profiler import LineProfiler
from PyBGEnv import qubic
import random
import numpy as np

print(qubic.action_num)
def test_valid_actions():
    state = qubic.init()
    player = 0
    actions = qubic.valid_actions(state, player)
    actions = qubic.valid_actions_np(state, player)

def test_game():
    state = qubic.init()
    player = 0
    while True:
        actions = qubic.valid_actions(state, player)
        action = qubic.minimax_action(state, player, 0)
        action = qubic.minimax_action(state, player, 1)
        action = qubic.minimax_action(state, player, 2)
        action = qubic.minimax_action(state, player, 3)
        action = random.choice(actions)
        next_state = qubic.get_next(state, action, player)
        state = next_state
        s = qubic.hash(state)
        p = qubic.current_player(state)
        if qubic.is_done(state, player):
            is_draw = qubic.is_draw(state)
            is_win = qubic.is_win(state, player)
            break
        player = 1 - player


prof = LineProfiler()
prof.add_function(test_game)
print("profiler start")
for i in range(1_000):
    prof.runcall(test_game)
prof.print_stats()
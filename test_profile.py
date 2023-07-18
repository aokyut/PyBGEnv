from line_profiler import LineProfiler
from PyBGEnv import qubic


def test_valid_actions():
    state = qubic.init()
    player = 0
    actions = qubic.valid_actions(state, player)
    actions = qubic.valid_actions_np(state, player)

prof = LineProfiler()
prof.add_function(test_valid_actions)
for i in range(100_000):
    prof.runcall(test_valid_actions)
prof.print_stats()
from PyBGEnv import qubic


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
state = qubic.init()
player = 0
show(state)
while True:
    actions = qubic.valid_actions(state, player)
    print(actions)
    
    a = int(input())
    next_state = qubic.get_next(state, a, player)
    state = next_state
    show(state)
    print(qubic.hash(state))
    if qubic.is_done(state, player):
        print("draw : ", qubic.is_draw(state))
        print("win : ", qubic.is_win(state, player))
        print(qubic.result(state))
        break
    player = 1 - player


env.action_num int
env.init() -> ndarray
env.current_player(ndarray state) -> int
env.get_next(ndarray state, int action, int player) -> ndarray
env.is_win(ndarray state, player) -> uint8(0 or 1)
env.is_draw(ndarray state) -> uint8(0 or 1)
env.is_done(ndarray state, player) -> uint8(0 or 1)
env.valid_action(ndarray state, player) -> ndarray[valid_acton_num]
env.result(ndarray state) -> ndarray[2]
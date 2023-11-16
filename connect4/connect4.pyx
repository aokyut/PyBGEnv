# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
import cython
import numpy as np
import random
cimport numpy as cnp

ctypedef cnp.uint8_t b_t
ctypedef unsigned char u8
ctypedef unsigned long long ull

action_num = 7
shift_1_mask = 0b0001111_0001111_0001111_0001111_0001111_0001111 
shift_6_mask = 0b0000000_1111000_1111000_1111000_1111000_1111000

cdef inline cnp.ndarray[b_t, ndim=1] copy(cnp.ndarray[b_t, ndim=1] b):
    cdef cnp.ndarray[b_t, ndim=1] b_
    cdef int i
    b_ = np.empty(84, dtype=np.uint8)
    for i in range(84):
        b_[i] = b[i]
    return b_

cpdef inline cnp.ndarray[b_t, ndim=1] init():
    cdef cnp.ndarray[b_t, ndim=1] b
    b = np.zeros(84, dtype=np.uint8)
    return b

cpdef inline int current_player(cnp.ndarray[b_t, ndim=1] b):
    cdef int i, sum_b = 0
    for i in range(84):
        sum_b += b[i]
    return sum_b % 2


cpdef inline cnp.ndarray[b_t, ndim=1] get_next(cnp.ndarray[b_t, ndim=1] b, int action, int player):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[b_t, ndim=1] stone
    cdef int pos
    next_b = copy(b)
    stone = b[0:42] + b[42:84]
    if stone[action + 14] == 0:
        if stone[action] == 0:
            pos = action 
        elif stone[action+7] == 0:
            pos = action + 7 
        else:
            pos = action + 14 
    elif stone[action + 21] == 0:
        pos = action + 21
    elif stone[action + 28] == 0:
        pos = action + 28
    else:
        pos = action + 35
    
    pos += player * 42
    next_b[pos] = 1
    return next_b


cpdef inline u8 is_win(cnp.ndarray[b_t, ndim=1] b, player):
    cdef cnp.ndarray[b_t, ndim=1] search_board
    cdef int i, bitboard
    if player == 0:
        for i in range(41, -1, -1):
            bitboard = (bitboard << 1) + b[i]
    else:
        for i in range(83, 41, -1):
            bitboard = (bitboard << 1) + b[i]
    if (bitboard & (bitboard >> 1) & (bitboard >> 2) & (bitboard >> 3) & shift_1_mask) > 0:
        return 1
    if (bitboard & (bitboard >> 7) & (bitboard >> 14) & (bitboard >> 21)) > 0:
        return 1
    if (bitboard & (bitboard >> 8) & (bitboard >> 16) & (bitboard >> 24) & shift_1_mask) > 0:
        return 1
    if (bitboard & (bitboard >> 6) & (bitboard >> 12) & (bitboard >> 18) & shift_6_mask) > 0:
        return 1
    return 0

cpdef inline u8 is_win_at(cnp.ndarray[b_t, ndim=1] b, int action, int player):
    pass

cpdef inline u8 is_draw(cnp.ndarray[b_t, ndim=1] b):
    cdef u8 s
    cdef int i
    for i in range(35, 42):
        s += b[i] + b[i + 42]
    if s == 14:
        return 1
    return 0

cpdef inline u8 is_done(cnp.ndarray[b_t, ndim=1] b, int player):
    return is_draw(b) | is_win(b, player)

cpdef inline cnp.ndarray[b_t, ndim=1] get_action_mask(cnp.ndarray[b_t, ndim=1] b):
    return b[35:42] + b[77:84]

cpdef inline cnp.ndarray[cnp.int64_t, ndim=1] valid_actions(cnp.ndarray[b_t, ndim=1] b, int player):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] tar
    cdef int i, j, size = 0
    cdef u8[7] mask
    for i in range(7):
        mask[i] = 1 - b[i + 35] - b[i + 77]
        size += mask[i]
    tar = np.empty(size, dtype=np.int64)

    j = 0
    for i in range(7):
        if mask[i] == 1:
            tar[j] = i
            j += 1
    return tar

cpdef inline cnp.ndarray[cnp.int64_t, ndim=1] result(cnp.ndarray[b_t, ndim=1] b):
    cdef int player
    player = current_player(b)
    if is_win(b, 1 - player):
        if player == 0:
            return np.array([-1, 1])
        else:
            return np.array([1, -1])
    else:
        return np.array([0, 0])

cpdef inline cnp.ndarray[b_t, ndim=1] flip(cnp.ndarray[b_t, ndim=1] b):
    cdef cnp.ndarray[b_t, ndim=1] _b
    cdef int i, j, k
    _b = np.zeros(84, dtype=np.uint8)
    for i in range(12):
        for j in range(7):
            _b[i * 7 + j] = b[i * 7 + 6 - j]
    return _b

cpdef inline str hash_unicode(cnp.ndarray[b_t, ndim=1] b):
    cdef str s = ""
    cdef int i
    cdef ull pack = 0

    for i in range(84):
        pack = (pack << 1) + b[i]
        if i % 7 == 6:
            s += chr(pack)
            pack = 0
    return s

cpdef inline str unique_hash(cnp.ndarray[b_t, ndim=1] b):
    cdef str s1 = ""
    cdef str s2 = ""
    cdef int i
    cdef cnp.ndarray[b_t, ndim=1] flipped
    
    s1 = hash_unicode(b)
    s2 = hash_unicode(flip(b))
    if s1 > s2:
        return s2
    return s1

cpdef inline str hash(cnp.ndarray[b_t, ndim=1] b):
    cdef str s = ""
    cdef int i
    cdef ull pack = 0
    for i in range(64):
        pack = (pack << 1) + b[i]
    pack = 0
    s += str(pack) + ","
    for i in range(64, 84):
        pack = (pack * 2) + b[i]
    s += str(pack)
    return s

cpdef inline u8 __is_invalid_action(cnp.ndarray[b_t, ndim=1] b, action):
    return b[action + 35] | b[action + 77]

cdef inline int __minimax(cnp.ndarray[b_t, ndim=1] b, int player, int rec, int depth):
    cdef int action, val, max_val = -2, i
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[cnp.int64_t, ndim=1] actions

    actions = valid_actions(b, player)
    for i in range(len(actions)):
        action = actions[i]
        #if __is_invalid_action(b, action): continue
        next_b = get_next(b, action, player)
        # 勝った時
        if is_win(next_b, player):
            val = 1
            return 1
        # 引き分けの時
        if is_draw(next_b):
            val = 0
        # どちらでも無い（勝敗が決まっていない時)
        else:
            if rec < depth:
                val = -__minimax(next_b, 1 - player, rec + 1, depth)
            else:
                val = 0
        if max_val < val:
            max_val = val
    
    return max_val

cpdef inline int minimax_action(cnp.ndarray[b_t, ndim=1] b, int player, int depth):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[cnp.int16_t, ndim=1] max_actions, vals
    cdef int action, max_val=-2, max_val_count=16, val, i
    vals = np.zeros(7, dtype=np.int16)
    
    for action in range(7):
        vals[action] = -2
        if __is_invalid_action(b, action): 
            vals[action] = -3
            continue
        next_b = get_next(b, action, player)
        if is_win(next_b, player): return action
        if is_draw(next_b): return action
        if depth == 0: continue
        val = -__minimax(next_b, 1-player, 0, depth - 1)
        if val == 1: return action
        if max_val < val:
            max_val = val
            max_val_count = 1
        elif max_val == val:
            max_val_count += 1
        vals[action] = val
    
    i = 0
    max_actions = np.zeros(max_val_count, dtype=np.int16)
    for action in range(7):
        if vals[action] == max_val:
            max_actions[i] = action
            i += 1
    
    return random.choice(max_actions)

cpdef inline cnp.ndarray[b_t, ndim=1] get_nonmate_actions(cnp.ndarray[b_t, ndim=1] b, int player, int depth):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[cnp.int16_t, ndim=1] max_actions, vals
    cdef int action, max_val=-2, max_val_count=16, val, i
    vals = np.zeros(7, dtype=np.int16)
    
    for action in range(7):
        vals[action] = -2
        if __is_invalid_action(b, action): 
            vals[action] = -3
            continue
        next_b = get_next(b, action, player)
        if is_win(next_b, player): 
            return np.array([action], dtype=np.int16)
        if is_draw(next_b): 
            return np.array([action], dtype=np.int16)
        if depth == 0: continue
        val = -__minimax(next_b, 1-player, 0, depth - 1)
        if val == 1: 
            return np.array([action], dtype=np.int16)
        if max_val < val:
            max_val = val
            max_val_count = 1
        elif max_val == val:
            max_val_count += 1
        vals[action] = val
    
    i = 0
    max_actions = np.zeros(max_val_count, dtype=np.int16)
    for action in range(7):
        if vals[action] == max_val:
            max_actions[i] = action
            i += 1
    
    return max_actions

cpdef inline int has_mate(cnp.ndarray[b_t, ndim=1] b, int player, int depth):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[cnp.int16_t, ndim=1] max_actions, vals
    cdef int action, max_val=-2, max_val_count=16, val, i
    vals = np.zeros(7, dtype=np.int16)
    
    for action in range(7):
        vals[action] = -2
        if __is_invalid_action(b, action): 
            vals[action] = -3
            continue
        next_b = get_next(b, action, player)
        if is_win(next_b, player): return action
        if is_draw(next_b): return action
        if depth != 0:
            val = -__minimax(next_b, 1-player, 0, depth - 1)
            if val == 1: return action
    
    return -1
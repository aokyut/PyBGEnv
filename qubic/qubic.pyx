# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=True, wraparound=True
# cython: cdivision=True
import cython
import numpy as np
cimport numpy as cnp
from cpython cimport array

ctypedef cnp.uint8_t b_t
ctypedef unsigned char u8
ctypedef unsigned long long ull

cdef int action_num = 16

cdef inline cnp.ndarray[b_t, ndim=1] copy(cnp.ndarray[b_t, ndim=1] b):
    cdef cnp.ndarray[b_t, ndim=1] b_
    cdef int i
    b_ = np.empty(128, dtype=np.uint8)
    for i in range(128):
        b_[i] = b[i]
    return b_

cpdef inline cnp.ndarray[b_t, ndim=1] init():
    cdef cnp.ndarray[b_t, ndim=1] b
    b = np.zeros(128, dtype=np.uint8)
    return b


cpdef inline int current_player(cnp.ndarray[b_t, ndim=1] b):
    cdef int i, sum_b = 0
    for i in range(128):
        sum_b += b[i]
    return sum_b % 2

cpdef inline cnp.ndarray[b_t, ndim=1] get_next(cnp.ndarray[b_t, ndim=1] b, int action, int player):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    cdef cnp.ndarray[b_t, ndim=1] stone
    cdef int pos
    next_b = copy(b)
    stone = b[0:64] + b[64:128]
    if stone[action] == 0:
        pos = action
    elif stone[action + 16] == 0:
        pos = action + 16
    elif stone[action + 32] == 0:
        pos = action + 32
    else:
        pos = action + 48
    pos += player * 64
    next_b[pos] = 1
    return next_b

@cython.boundscheck(False)
cpdef inline u8 is_win(cnp.ndarray[b_t, ndim=1] b, player):
    cdef int i, n, m, l
    cdef cnp.ndarray[b_t, ndim=1] tar_arr, x, y, z, xy, yx, xz, zx, yz, zy
    x = np.zeros(16, dtype=np.uint8)
    y = np.zeros(16, dtype=np.uint8)
    z = np.zeros(16, dtype=np.uint8)
    xy = np.zeros(4, dtype=np.uint8)
    yx = np.zeros(4, dtype=np.uint8)
    xz = np.zeros(4, dtype=np.uint8)
    zx = np.zeros(4, dtype=np.uint8)
    yz = np.zeros(4, dtype=np.uint8)
    zy = np.zeros(4, dtype=np.uint8)
    if player == 0:
        tar_arr = b[0:64]
    else:
        tar_arr = b[64: 128]
    for i in range(64):
        n = i // 16
        l = i % 4
        m = ((i % 16) // 4)
        if tar_arr[i] == 0: continue
        x[4*m + l] += 1
        y[4*n + l] += 1
        z[4*n + m] += 1
        if n == m:
            xy[l] += 1
        elif n + m == 4:
            yx[l] += 1
        if n == l:
            xz[m] += 1
        elif n + l == 4:
            zx[m] += 1
        if m == l:
            yz[n] += 1
        elif l + m == 4:
            zy[n] += 1
    for i in range(16):
        if x[i] == 4: return 1
        if y[i] == 4: return 1
        if z[i] == 4: return 1
    for i in range(4):
        if xy[i] == 4: return 1
        if yx[i] == 4: return 1
        if xz[i] == 4: return 1
        if zx[i] == 4: return 1
        if yz[i] == 4: return 1
        if zy[i] == 4: return 1
    if (tar_arr[0] & tar_arr[21] & tar_arr[42] & tar_arr[63]) == 1:
        return 1
    if (tar_arr[3] & tar_arr[22] & tar_arr[41] & tar_arr[60]) == 1:
        return 1
    if (tar_arr[12] & tar_arr[25] & tar_arr[38] & tar_arr[51]) == 1:
        return 1
    if (tar_arr[15] & tar_arr[26] & tar_arr[37] & tar_arr[48]) == 1:
        return 1
    return 0

cpdef inline u8 is_draw(cnp.ndarray[b_t, ndim=1] b):
    cdef u8 s
    for i in range(128):
        s += b[i]
    if s == 64:
        return 1
    return 0

cpdef inline u8 is_done(cnp.ndarray[b_t, ndim=1] b, int player):
    return is_draw(b) | is_win(b, player)

cpdef inline cnp.ndarray[b_t, ndim=1] get_action_mask(cnp.ndarray[b_t, ndim=1] b):
    return b[48:64]

# validでないactionは-1の配列[a1, a2, a3, ... , ai, -1, -1, ...]
cpdef inline cnp.ndarray[b_t, ndim=1] valid_actions(cnp.ndarray[b_t, ndim=1] b, int player):
    cdef cnp.ndarray[b_t, ndim=1] tar, stone
    cdef int i, j, size = 0
    stone = b[0:64] + b[64:128]
    for i in range(48, 64):
        if stone[i] == 0:
            size += 1
    tar = np.empty(size, dtype=np.uint8)
    j = 0
    for i in range(16):
        if stone[i + 48] == 0:
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

cpdef inline str hash(cnp.ndarray[b_t, ndim=1] b):
    cdef str s = ""
    cdef int i
    cdef ull pack = 0
    
    for i in range(128):
        pack = (pack << 1) + b[i]
        if i % 64 == 63:
            s += str(pack)
            s += ","
            pack = 0
        #s += str(b[i])
    return s
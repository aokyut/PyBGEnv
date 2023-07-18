# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
import cython
import numpy as np
cimport numpy as cnp

ctypedef cnp.uint8_t b_t

# 18要素の配列を出力
cpdef inline cnp.ndarray[b_t, ndim=1] init():
    cdef cnp.ndarray[b_t, ndim=1] b
    b = np.zeros(18, dtype=np.uint8)
    return b

cpdef inline int current_player(cnp.ndarray[b_t, ndim=1] b):
    cdef int sum_b
    sum_b = b.sum()
    return sum_b % 2

cpdef inline cnp.ndarray[b_t, ndim=1] get_next(cnp.ndarray[b_t, ndim=1] b, int action, int player):
    cdef cnp.ndarray[b_t, ndim=1] next_b
    next_b = init()
    for i in range(18):
        next_b[i] = b[i]
    if player == 0:
        next_b[action] = 1
    else:
        next_b[action + 9] = 1
    return next_b

cpdef inline unsigned char is_win(cnp.ndarray[b_t, ndim=1] b, int player):
    if player == 0:
        return (b[0] & b[1] & b[2] | 
               b[0] & b[3] & b[6] | 
               b[0] & b[4] & b[8] |
               b[1] & b[4] & b[7] |
               b[2] & b[4] & b[6] |
               b[2] & b[5] & b[8] |
               b[3] & b[4] & b[5] |
               b[6] & b[7] & b[8])
    else:
        return (b[9] & b[10] & b[11] | 
               b[9] & b[12] & b[15] | 
               b[9] & b[13] & b[17] |
               b[10] & b[13] & b[16] |
               b[11] & b[13] & b[15] |
               b[11] & b[14] & b[17] |
               b[12] & b[13] & b[14] |
               b[15] & b[16] & b[17])

cpdef inline unsigned char is_draw(cnp.ndarray[b_t, ndim=1] b):
    if b.sum() == 9:
        return 1
    return 0

cpdef inline unsigned char is_done(cnp.ndarray[b_t, ndim=1] b, int player):
    return is_draw(b) | is_win(b, player)

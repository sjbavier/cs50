"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    played = 0
    for row in board:
        for column in row:
            if column != None:
                played += 1
    
    if played % 2 == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i, row in enumerate( board ):
        for j, column in enumerate(row):
            if column != EMPTY:
                actions.add((i, j))

    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    players_mark = player(board)
    if board[i][j] != EMPTY:
        print(f"invalid move {i} {j}")
        raise ValueError
    else:
        copy_board = [row[:] for row in board] # deep copy list comprehension for small matrix
        copy_board[i][j] = players_mark
        return copy_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    def check_hor():
        for row in board:
            row_set = set(row)
            if len(row_set) == 1 and row_set[0] != EMPTY:
                return row_set[0]
        return None

    def check_ver():
        transposed_board = list(zip(*board)) # trick to flip rows to cols
        for col in transposed_board:
            col_set = set(col)
            if len(col_set) == 1 and col_set[0] != EMPTY:
                # all 3 values in col are same and not EMPTY so return winner
                return col_set[0]
        return None

    def check_dia():
        diag_top = set()
        diag_bot = set()
        for i, row in enumerate(board):
            diag_top.add(board[i][i])
            diag_bot.add(board[i][len(board) - i])
        if len(diag_top) == 1 and diag_top[0] != EMPTY:
            return diag_top[0]
        if len(diag_bot) == 1 and diag_bot[0] != EMPTY:
            return diag_bot[0]
        return None
    
    def check_all(*funcs):
        for func in funcs:
            result = func()
            if result is not None:
                return result
        return None
    
    checks = [check_hor, check_ver, check_dia]
    result = check_all(checks)
    print(f"winner {result}")
    return result

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    is_winner = winner(board)
    # if there is a winner its over
    if is_winner is not None:
        return True
    # if there are actions left its not over
    if len(actions(board)) != 0:
        return False
    else:
        return True



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winning_player = winner(board)
    if winning_player is not None:
        if winning_player == X:
            return 1
        else:
            return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    

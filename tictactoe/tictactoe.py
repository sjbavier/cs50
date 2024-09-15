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
    for i, row in enumerate(board):
        for j, column in enumerate(row):
            if column == EMPTY:
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
        copy_board = [row[:] for row in board]  # deep copy list comprehension for small matrix
        copy_board[i][j] = players_mark
        return copy_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    def check_hor():
        for row in board:
            if all(col is not None for col in row) and len(set(row)) == 1:
                return set(row).pop()
        return None

    def check_ver():
        transposed_board = list(zip(*board))  # trick to flip rows to cols
        for column in transposed_board:
            if all(col is not None for col in column) and len(set(column)) == 1:
                return set(column).pop()
        return None

    def check_dia():
        diag_top = []
        diag_bot = []
        for i, row in enumerate(board):
            len_board_diag = len(board) - 1 - i
            diag_top.append(board[i][i])
            diag_bot.append(board[i][len_board_diag])
        if all(diag is not None for diag in diag_top) and len(set(diag_top)) == 1:
            return set(diag_top).pop()
        if all(diag is not None for diag in diag_bot) and len(set(diag_bot)) == 1:
            return set(diag_bot).pop()
        return None

    def check_all(*funcs):
        for func in funcs:
            result = func()
            if result is not None:
                return result
        return None

    checks = [check_hor, check_ver, check_dia]
    result = check_all(*checks)
    return result


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    is_winner = winner(board)
    # if there is a winner its over
    if is_winner is not None:
        return True
    # if there are actions left it's not over
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

    def max_value(state):
        if terminal(state):
            return utility(state)
        v = -math.inf
        for move in actions(state):
            new_r = result(state, move)
            v = max(v, min_value(new_r))

        return v

    def min_value(state):
        if terminal(state):
            return utility(state)
        v = math.inf
        for move in actions(state):
            new_r = result(state, move)
            v = min(v, max_value(new_r))
        return v

    # start initial values based on player
    current_player = player(board)
    best_action = None
    value = None
    if current_player == X:
        best_score = -math.inf
    else:
        best_score = math.inf

    # loop through initial actions
    for action in actions(board):
        new_result = result(board, action)
        if current_player == X:
            score = min_value(state=new_result)
            if score > best_score:
                best_action = action
                best_score = score
        if current_player == O:
            score = max_value(state=new_result)
            if score < best_score:
                best_action = action
                best_score = score

    return best_action

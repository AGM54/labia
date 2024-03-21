import numpy as np
import random
import sys
import math
from QLearner import QAgent,check_connection
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
GREEN = (0, 0, 255)
ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return (board[ROW_COUNT-1][col] == 0).any()

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_best_move(board, piece):

    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col

board = create_board()
game_over = False

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)


isdraw = True




# Inicialización del agente Q-learning
import matplotlib.pyplot as plt

state_size = ROW_COUNT * COLUMN_COUNT
action_size = COLUMN_COUNT
q_agent = QAgent(state_size, action_size)
turn = 0
total_rewards = []
#training the model against himself
for _ in range(50):
    board = create_board()
    game_over = False
    isdraw = True
    rewards = 0
    while not game_over:
        if not game_over:
            #TURN FORTHE DEEP QLEARNER
            # Representación del estado (aplanar el tablero)
            state = np.reshape(board, [1, state_size])

            # Selección de la acción por el agente
            col = q_agent.act(state)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                if turn == PLAYER:
                    drop_piece(board, row, col, PLAYER_PIECE)
                else: # its the other piece turn 
                    drop_piece(board, row, col, AI_PIECE)
                next = np.reshape(board, [1, state_size])
                if turn == PLAYER:
                    if check_connection(board ,row, col,PLAYER_PIECE,COLUMN_COUNT,ROW_COUNT): #reward for conecting pieces
                        q_agent.train(state,col,0.5,next,False)
                        rewards += 0.5
                else: # its the other piece turn 
                    if check_connection(board ,row, col,AI_PIECE,COLUMN_COUNT,ROW_COUNT): #reward for conecting pieces
                        q_agent.train(state,col,0.5,next,False)
                        rewards += 0.5
                if winning_move(board, PLAYER_PIECE): #big reward for winning move
                    q_agent.train(state,col,1,next,True)
                    game_over = True
                    isdraw = False
                    rewards += 1
            turn += 1
            turn = turn%2
    total_rewards.append(rewards)

plt.plot(total_rewards)
plt.xlabel('Episodio')
plt.ylabel('Recompensa acumulada')
plt.title('Recompensas acumuladas por episodio')
plt.grid(True)

# Guardar la gráfica en un archivo PDF
plt.savefig('recompensas.pdf')

# Mostrar la gráfica
plt.show()
                    
results = []
for _ in range(50):
    board = create_board()
    game_over = False
    isdraw = True
    while not game_over:
        if not game_over:
            #TURN FORTHE DEEP QLEARNER
            # Representación del estado (aplanar el tablero)
            state = np.reshape(board, [1, state_size])

            # Selección de la acción por el agente
            col = q_agent.act(state)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    game_over = True
                    isdraw = False
                    results.append(PLAYER_PIECE)
        if not game_over: #TRUN OF THE MIN MAX
            col, ai1_minimax_score = minimax(board, 5, -math.inf, math.inf, False)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    results.append(AI_PIECE)
                    game_over = True
                    isdraw = False
        if game_over:
            if isdraw:
                results.append(0)
print(results)

draws = results.count(0)
qlearner_wins = results.count(1)
minmax_wins = results.count(2)

# Gráfico de barras
labels = ['Empate', 'Qlearner Wins', 'Minimax Wins']
values = [draws, qlearner_wins, minmax_wins]

plt.bar(labels, values, color=['blue', 'green', 'red'])
plt.xlabel('Resultados')
plt.ylabel('Cantidad de juegos')
plt.title('Resultados de las iteraciones')
plt.grid(True)

# Guardar la gráfica de barras en un archivo PDF
plt.savefig('resultados_barras.pdf')

# Mostrar la gráfica de barras
plt.show()

# Gráfico de resultados por iteración
plt.plot(results)
plt.xlabel('Iteración')
plt.ylabel('Resultado (0: Empate, 1: Qlearner Wins, 2: Minimax Wins)')
plt.title('Resultados por iteración')
plt.grid(True)

# Guardar la gráfica de resultados por iteración en un archivo PDF
plt.savefig('resultados_iteracion.pdf')

# Mostrar la gráfica de resultados por iteración
plt.show()
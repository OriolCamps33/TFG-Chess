import chess
from Q_Learn import *
from NNG import *
from tqdm import tqdm

def agents_game(agent1, agent2, games):
    white_wins = 0
    black_wins = 0
    for _ in tqdm(range(games), desc="Chess Games: "):
        board = chess.Board()
        running = True
        trun_white = True
        while running:
            if not board.is_game_over():
                if trun_white:
                    legal_moves = agent1.get_legal_moves(board)
                    action = agent1.choose_action(board, legal_moves)                
                    board.push(action)
                    trun_white = False
                else:
                    state = str(board)
                    legal_moves = agent2.get_legal_moves(board)
                    action = agent2.choose_action(state, legal_moves)                
                    board.push(action)
                    trun_white = True
            else:
                result = board.outcome()
                if result.winner == chess.WHITE:
                    white_wins += 1
                elif result.winner == chess.BLACK:
                    black_wins += 1
                
                running = False
    return [white_wins, black_wins]


if __name__ == "__main__":
    black = Q_Learn_Agent()
    print("\nLoading Q_Learn Model")
    black.loadModel("M_1,5M.txt")

    white = NNG_Agent()
    print("\nLoading NNG Model")
    white.load("Models/NNG/E_1000.h5")

    wins = agents_game(white, black, 200)
    print("\n\nWhite wins: " + str(wins[0]))
    print("Black wins: " + str(wins[1]))



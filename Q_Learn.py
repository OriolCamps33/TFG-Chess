import chess
import chess.engine
import random
from tqdm import tqdm
from ChessGame import *
from multiprocessing import Pool


class ChessAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.q_values = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        self.file_path = "Models/"


    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)
    

    def choose_action(self, state, legal_moves):
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
        else:
            max_q_value = float('-inf')  # Set an initial minimum value
            best_moves = []
            for move in legal_moves:
                q_value = self.get_q_value(state, move)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_moves = [move]  # Restart the list for ties
                elif q_value == max_q_value:
                    best_moves.append(move)
            return random.choice(best_moves)
        

    def update_q_value(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)
        legal_next_moves = self.get_legal_moves(next_state)
        best_next_q_value = max([self.get_q_value(str(next_state), next_action) for next_action in legal_next_moves])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * best_next_q_value - old_q_value)
        self.q_values[(state, action)] = new_q_value


    def get_legal_moves(self, board):
        return list(board.legal_moves)
    

    def train(self, iter=1000):
        high_reward = 0.0
        low_reward = float('inf')
        for _ in tqdm(range(iter), desc="Iterations: "):
            board = chess.Board()
            total_reward = 0

            while not board.is_game_over():
                
                state = str(board)
                legal_moves = self.get_legal_moves(board)
                action = self.choose_action(state, legal_moves)
                
                board.push(action)
                next_state = board
                reward = 0.5

                if len(board.move_stack) >= 2:
                    
                    last_move = str(board.peek())  # Casilla después del último movimiento
                    pre_move = board.pop()

                    # Obtener las posiciones antes y después del último movimiento
                    pos_aft = chess.parse_square(last_move[2:4])

                    # Obtener las piezas en las posiciones antes y después del movimiento
                    pieza_aft = board.piece_at(pos_aft)

                    if pieza_aft is not None:
                        reward = pieza_aft.piece_type
                        self.update_q_value(state, action, reward, next_state)
                    
                    board.push(pre_move)

                # Reward based on game outcome (combine checks)
                if board.is_checkmate():
                    reward = 10
                elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
                    reward = 0

                total_reward += reward
            if(total_reward < low_reward): low_reward = total_reward
            if(total_reward > high_reward): high_reward = total_reward

            self.exploration_rate *= self.exploration_decay
            #print("Total reward in this game:", total_reward)
        print("Hight reward: " + str(high_reward))
        print("Low reward: " + str(low_reward))
    
    def saveModel(self, filename):
        file = open(self.file_path + filename, "w")
        file.write(str(self.q_values))
        file.write(";")

        file.write(str(self.learning_rate))
        file.write(";")

        file.write(str(self.discount_factor))
        file.write(";")

        file.write(str(self.exploration_rate))
        file.write(";")

        file.write(str(self.exploration_decay))

        print("Model saved in " + self.file_path + filename)
        print()

    def loadModel(self, filename):
        file = open(self.file_path + filename, "r")
        data = file.readline().split(";")

        q_values = data[0].split(",")
        q_values[0] = q_values[0][1:]
        q_values[-1] = q_values[-1][:-1]
        for i in range(0, len(q_values), 2):
            q = q_values[i] + "," + q_values[i+1]
            d = q.split(":")
            self.q_values[d[0]] = d[1]


        self.learning_rate = float(data[1])
        self.discount_factor = float(data[2])
        self.exploration_rate = float(data[3])
        self.exploration_decay = float(data[4])
        print(type(self.q_values))

        



if __name__ == "__main__":
    agent = ChessAgent()
    agent.loadModel('M_10k.txt')
    
    game = ChessGame()
    game.game(agent)



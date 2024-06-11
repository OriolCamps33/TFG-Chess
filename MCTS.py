import chess
import numpy as np
import random
import tensorflow as tf

class ChessMCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_maximizing(self):
        return self.state.turn == chess.WHITE

    def expand(self):
        legal_moves = list(self.state.legal_moves)
        for move in legal_moves:
            new_state = self.state.copy()
            new_state.push(move)
            self.children[move] = ChessMCTSNode(new_state, parent=self)

    def select_child(self, exploration_constant=1.0):
        return max(self.children.values(), key=lambda child: child.get_ucb_value(exploration_constant))

    def get_ucb_value(self, exploration_constant):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

class ChessMCTS:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def search(self, state, num_simulations=50):
        root = ChessMCTSNode(state)

        for x in range(num_simulations):
            leaf = self.traverse(root)
            value = self.evaluate(leaf)
            leaf.backpropagate(value)

        best_move = max(root.children.keys(), key=lambda move: root.children[move].visits)
        return best_move

    def traverse(self, node):
        while not node.is_leaf():
            node = node.select_child()


        if node.visits > 0:  # Ensure we don't expand a terminal node
            node.expand()

            node = random.choice(list(node.children.values()))
        return node

    def boardstate(self, fen):
        try:
            board = chess.Board(str(fen))
        except ValueError:
            print("Error: Debes introducir un número válido.")
            return 0

        def castling_rights(color):
            return [
                board.has_kingside_castling_rights(color),
                board.has_queenside_castling_rights(color),
                board.is_check() if color == chess.WHITE else board.was_into_check()
            ]

        WCKI, WCQ, WCH = map(int, castling_rights(chess.WHITE))
        BCKI, BCQ, BCH = map(int, castling_rights(chess.BLACK))
        fw, fb = [WCKI, WCQ, WCH], [BCKI, BCQ, BCH]

        piece_map = {
            'p': -1, 'n': -3, 'b': -4, 'r': -5, 'q': -9, 'k': -100,
            'P': 1, 'N': 3, 'B': 4, 'R': 5, 'Q': 9, 'K': 100, '.': 0
        }

        bstr = [piece_map[c] for row in str(board).split('\n') for c in row.replace(' ', '')]

        if 'w' not in str(fen):
            bstr = [-v for v in reversed(bstr)]
            fw, fb = fb, fw

        BITBOARD = fw + fb + bstr
        return BITBOARD

    def board_to_input(self, board_fen):
        info_board = self.boardstate(board_fen)

        inputmeta = info_board[:6]
        inputboard = info_board[6:]

        inputboard = np.array(inputboard).reshape((1, -1))  # Adjust shape as needed
        inputmeta = np.array(inputmeta).reshape((1, -1))    # Adjust shape as needed

        return inputboard, inputmeta

    def evaluate(self, node):
        ib, im = self.board_to_input(node.state.fen())
        input_data = [ib, im]
        value = self.neural_network.predict(input_data, verbose=0)
        return value


if __name__ == "__main__":
  model = tf.keras.models.load_model("Models/NNG/model_400.keras")
  board = chess.Board()

  mcts_agent = ChessMCTS(model)
  best_move = mcts_agent.search(board)
  print("Mejor movimiento encontrado:", best_move)
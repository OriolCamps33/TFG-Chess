import chess
import chess.engine


class Leela:
    def __init__(self, url, depth=5) -> None:

        self.engine = chess.engine.SimpleEngine.popen_uci(url)
        self.limit = chess.engine.Limit(depth=depth)

    def get_legal_moves(self, board):

        return list(board.legal_moves)
    
    def choose_action(self, board, legal_moves):
        result = self.engine.play(board, self.limit)
        return result.move

        
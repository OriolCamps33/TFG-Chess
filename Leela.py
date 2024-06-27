import chess
import chess.engine


class Leela:
    def __init__(self, url_exe, url_network, depth=5) -> None:

        self.engine = chess.engine.SimpleEngine.popen_uci([url_exe, f'--weights={url_network}'])
        self.limit = chess.engine.Limit(depth=depth)

    def get_legal_moves(self, board):

        return list(board.legal_moves)
    
    def choose_action(self, board, legal_moves):
        result = self.engine.play(board, self.limit)
        return result.move

if __name__ == "__main__":
    eng = Leela("Models/LC0/lc0-v0.30.0-windows/lc0.exe", "Models/LC0/lc0-v0.30.0-windows/791556.pb.gz")


        
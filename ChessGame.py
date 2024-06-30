import chess
import pygame
import sys
import time
from Pythons.NNG import NNG_Agent
from Pythons.Stockfish import Stockfish
from Pythons.Leela import Leela
from Pythons.MCTS import MCTS
import random

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.width = 800
        self.height = 800
        self.dimension = 8
        self.sq_size = self.height // self.dimension
        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
        self.images = {
            'r': pygame.image.load('Imagenes/B_R.png'),
            'n': pygame.image.load('Imagenes/B_N.png'),
            'b': pygame.image.load('Imagenes/B_B.png'),
            'q': pygame.image.load('Imagenes/B_Q.png'),
            'k': pygame.image.load('Imagenes/B_K.png'),
            'p': pygame.image.load('Imagenes/B_P.png'),
            'R': pygame.image.load('Imagenes/W_R.png'),
            'N': pygame.image.load('Imagenes/W_N.png'),
            'B': pygame.image.load('Imagenes/W_B.png'),
            'Q': pygame.image.load('Imagenes/W_Q.png'),
            'K': pygame.image.load('Imagenes/W_K.png'),
            'P': pygame.image.load('Imagenes/W_P.png')
        }
    
    def move(self, ini_pos, fin_pos):
        if ini_pos == fin_pos:
            print("Movimiento no valido")
            return False
        str_ini_pos = chr(96 + ini_pos[1]) + str(ini_pos[0])
        str_fin_pos = chr(96 + fin_pos[1]) + str(fin_pos[0])
        
        move = str_ini_pos + str_fin_pos
        
        m = chess.Move.from_uci(move)
        
        if str_fin_pos[1] == '8' and self.board.piece_type_at(m.from_square) == chess.PAWN:
            m.promotion = chess.QUEEN

        if m in list(self.board.legal_moves):
            self.board.push(m)
            return True
        else:
            print("Movimiento no valido")
            return False
    
    def print_tab(self, screen):
        for row in range(self.dimension):
            for col in range(self.dimension):
                color = self.light_color if (row + col) % 2 == 0 else self.dark_color
                pygame.draw.rect(screen, color, pygame.Rect(col*self.sq_size, row*self.sq_size, self.sq_size, self.sq_size))

    def draw_pieces(self, board, screen):
        for row in range(self.dimension):
            for col in range(self.dimension):
                piece = self.board_to_matrix()[row][col]
                if piece != "-":
                    screen.blit(self.images[piece], pygame.Rect(col*self.sq_size+12, row*self.sq_size+12, self.sq_size, self.sq_size))
    
    def board_to_matrix(self):
        matrix = [['-' for _ in range(8)] for _ in range(8)]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                piece_symbol = piece.symbol()
                matrix[7 - chess.square_rank(square)][chess.square_file(square)] = piece_symbol
        return matrix
    
    def get_row_col_from_mouse(self, pos):
        x, y = pos
        row = y // self.sq_size
        row = 8 - row
        col = x // self.sq_size +1
        return row, col

    def game(self, agent):
        # Crear la ventana
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tablero de Ajedrez")

        running = True
        selected_piece = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # Si se ha presionado el botón izquierdo del mouse
                        mouse_pos = pygame.mouse.get_pos()
                        
                        row, col = self.get_row_col_from_mouse(mouse_pos)
                        if not selected_piece:
                            selected_piece = True
                            selected_square = (row, col)
                        else:
                            move_maked = self.move(selected_square, (row, col))
                            selected_piece = False
                            selected_square = None
                            self.print_tab(screen)
                            self.draw_pieces(self.board, screen)
                            pygame.display.flip()

                            if(move_maked and not self.board.is_game_over()):
                                legal_moves = agent.get_legal_moves(self.board)
                                action = agent.choose_action(self.board, legal_moves)           
                                self.board.push(action)

            
            self.print_tab(screen)
            self.draw_pieces(self.board, screen)
            pygame.display.flip()
            pygame.time.Clock().tick(60)

        pygame.quit()
        sys.exit()   

    def agents_game_vista(self, agent1, agent2):
        # Crear la ventana
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tablero de Ajedrez")

        self.print_tab(screen)
        self.draw_pieces(self.board, screen)
        pygame.display.flip()
        pygame.time.Clock().tick(60)

        trun_white = True
        while not self.board.is_game_over():
            if trun_white:
                legal_moves = agent1.get_legal_moves(self.board)
                action = agent1.choose_action(self.board, legal_moves)                
                self.board.push(action)
                trun_white = False
            else:
                legal_moves = agent2.get_legal_moves(self.board)
                action = agent2.choose_action(self.board, legal_moves)                
                self.board.push(action)
                trun_white = True
            
            self.print_tab(screen)
            self.draw_pieces(self.board, screen)
            pygame.display.flip()
            pygame.time.Clock().tick(60)

        pygame.quit()
        sys.exit() 

    def agents_game(self, agent1, agent2):

        board = chess.Board()
        trun_white = True
        turns = 0
        while not board.is_game_over():
            turns += 1
            if trun_white:
                legal_moves = agent1.get_legal_moves(board)
                action = agent1.choose_action(board, legal_moves)
                board.push(action)
                trun_white = False
            else:
                legal_moves = agent2.get_legal_moves(board)
                action = agent2.choose_action(board, legal_moves)
                board.push(action)
                trun_white = True

        print(turns)
        return board.result()


    def competi(self, agent1, agent2, num_match):
        results = []
        for x in range(num_match):
            print("Init Game", x)
            white = random.randint(1, 2)
            if white == 1:
                result = self.agents_game(agent1=agent1, agent2=agent2)
            else:
                result = self.agents_game(agent1=agent2, agent2=agent1)

            res = 1
            if result == None:
                res = 0
            elif result == "0-1":
                res = -1
            
            if white != 1:
                res *= -1
            
            results.append(res)

        print(results)


if __name__ == "__main__":
    nng = NNG_Agent("Pythons/Models/NNG/model_200_model_gran.h5")
    #stf = Stockfish("Pythons/Models/stockfish/stockfish-windows-x86-64-avx2.exe")
    mcts = MCTS("Pythons/Models/NNG/model_200_model_gran.h5", 500)
    #lc0 = Leela("Pythons/Models/LC0/lc0.exe", "Pythons/Models/LC0/791556.pb.gz")
    
    game = ChessGame()
    print("NNG vs MCTS")
    game.competi(nng, mcts, 51)

    #game.game(mcts)
    
    print("end")


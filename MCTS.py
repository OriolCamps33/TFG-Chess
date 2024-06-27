
import tensorflow as tf
import numpy as np
import chess
import random
import pandas as pd
import math
from collections import defaultdict
from NNG import NNG_Agent
import time

class MCTS:
    def __init__(self, filepath, iteraciones=5) -> None:
        self.transposition_table = self.TranspositionTable()
        self.model = NNG_Agent(filepath)
        self.iteraciones = iteraciones
        self.table_times = 0

    class NodoMCTS:
        def __init__(self, tablero, move=None, parent=None):
            self.tablero = tablero
            self.move = move
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = list(tablero.legal_moves)

        def expand(self):
            # Generar posibles estados sucesores
            
            # Crear nodos hijos y añadir al árbol
            for move in self.untried_moves:
                nuevo_tablero = self.tablero.copy()
                nuevo_tablero.push(move)
                child_node = MCTS.NodoMCTS(nuevo_tablero, move, self)
                self.children.append(child_node)
            
            # Retornar uno de los nuevos nodos para la fase de simulación
            return random.choice(self.children) if self.children else None

        def select_child(self):
            # Utilizar UCB1 para seleccionar el nodo
            C = 1.4  # Constante de exploración
            return max(self.children, key=lambda c: c.wins)

        def update(self, result):
            self.visits += 1
            self.wins += result

    class TranspositionTable:
        def __init__(self):
            self.table = defaultdict(lambda: None)

        def lookup(self, fen):
            return self.table[fen]

        def store(self, fen, value):
            self.table[fen] = value
            
    def ordenar_movimientos(self, tablero):
        # Ordenar por capturas primero
        return sorted(tablero.legal_moves, key=lambda move: tablero.is_capture(move), reverse=True)

    def evaluar_posicion(self, board):
        evaluacion = self.model.predict(board)
        return evaluacion 

    def evaluar_minimax(self, tablero, profundidad, alpha, beta, maximizando):
        fen = tablero.fen()
        hash_entry = self.transposition_table.lookup(fen)
        if hash_entry is not None:
            self.table_times += 1
            return hash_entry

        if profundidad == 0 or tablero.is_game_over():
            evaluacion = self.evaluar_posicion(tablero)
            self.transposition_table.store(fen, evaluacion)
            return evaluacion
        
        if maximizando:
            max_eval = -float('inf')
            for move in self.ordenar_movimientos(tablero):
                tablero.push(move)
                evaluacion = self.evaluar_minimax(tablero, profundidad - 1, alpha, beta, False)
                tablero.pop()
                max_eval = max(max_eval, evaluacion)
                alpha = max(alpha, evaluacion)
                if beta <= alpha:
                    break
            self.transposition_table.store(fen, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.ordenar_movimientos(tablero):
                tablero.push(move)
                evaluacion = self.evaluar_minimax(tablero, profundidad - 1, alpha, beta, True)
                tablero.pop()
                min_eval = min(min_eval, evaluacion)
                beta = min(beta, evaluacion)
                if beta <= alpha:
                    break
            self.transposition_table.store(fen, min_eval)
            return min_eval

    def choose_action(self, tablero, list_moves):
        root = self.NodoMCTS(tablero)
        for _ in range(self.iteraciones):
            node = root
            tablero_simulacion = tablero.copy()

            # Fase de selección
            while node.children and not tablero_simulacion.is_game_over():
                node = node.select_child()
                tablero_simulacion.push(node.move)

            # Fase de expansión
            if not tablero_simulacion.is_game_over():
                node = node.expand()

            # Fase de simulación
            result = self.evaluar_posicion(tablero_simulacion)
            #result = self.evaluar_minimax(tablero_simulacion, 3, -float('inf'), float('inf'), tablero_simulacion.turn == chess.WHITE)
            #print("Table times: ", self.table_times)
            if tablero.turn == chess.BLACK: result *= -1
            # Fase de retropropagación
            while node:
                node.update(result)
                node = node.parent

        return max(root.children, key=lambda c: c.wins).move
    
    def imprimir_arbol(self, nodo, nivel=0):
        print(' ' * (nivel * 4) + str(nodo.wins))
        for hijo in nodo.children:
            self.imprimir_arbol(hijo, nivel + 1)

    def get_legal_moves(self, board):
        return list(board.legal_moves)

if __name__ == "__main__":
    board = chess.Board()
    mcts_agent = MCTS("Models/NNG/model_200_model_gran.h5", 10)
    best_move = mcts_agent.choose_action(board, [])

    print("Mejor movimiento encontrado:", best_move)

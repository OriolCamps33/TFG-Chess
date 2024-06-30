
import tensorflow as tf
import numpy as np
import chess
import random
import time
from collections import defaultdict
from Pythons.NNG import NNG_Agent
import networkx as nx
import matplotlib.pyplot as plt

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
            self.valor = 0
            self.untried_moves = list(tablero.legal_moves)

        def expand(self):
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
            if self.tablero.turn == chess.WHITE:
                return max(self.children, key=lambda c: c.valor)
            else:
                return min(self.children, key=lambda c: c.valor)


        def update(self, result):
            self.valor += result

    def evaluar_posicion(self, board):
        evaluacion = self.model.predict(board)
        return evaluacion 
    
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
        move = chess.Move.from_uci("0000")
        root = self.NodoMCTS(tablero, move=move)

        for _ in range(self.iteraciones):
            node = root
            tablero_simulacion = tablero.copy()

            # Fase de selección
            while node.children and not tablero_simulacion.is_game_over():
                node = node.select_child()
                tablero_simulacion.push(node.move)

            # Fase de expansión
            if not tablero_simulacion.is_game_over():
                node.expand()

            # Fase de simulación
            #result = int(self.evaluar_posicion(tablero_simulacion))
            result = self.evaluar_minimax(tablero_simulacion, 5, float("inf"), -float("inf"), tablero_simulacion.turn)
            
            # Fase de retropropagación
            while node:
                node.update(result)
                node = node.parent
            
        #self.imprimir_arbol(root)
        if tablero.turn == chess.WHITE:
            return max(root.children, key=lambda c: c.valor).move
        else:
            return min(root.children, key=lambda c: c.valor).move
    
    # Función para crear un layout personalizado en forma de árbol vertical
    def tree_layout(self, grafo, root=None):
        if root is None:
            root = list(grafo.nodes)[0]
        
        def _hierarchy_pos(g, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
            pos = {root: (xcenter, vert_loc)}
            neighbors = list(g.neighbors(root))
            if not neighbors:
                return pos

            dx = width / len(neighbors) 
            nextx = xcenter - width / 2 - dx / 2
            for neighbor in neighbors:
                nextx += dx
                pos.update(_hierarchy_pos(g, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx))
            return pos

        return _hierarchy_pos(grafo, root)
    
    def imprimir_arbol(self, nodo):
        # Función para agregar nodos y aristas al grafo
        def agregar_nodos_y_aristas(nodo, grafo):
            if nodo.parent:
                grafo.add_edge(nodo.parent.move.uci(), nodo.move.uci())
            for hijo in nodo.children:
                agregar_nodos_y_aristas(hijo, grafo)

        # Crear el grafo
        grafo = nx.DiGraph()  # DiGraph para un grafo dirigido
        agregar_nodos_y_aristas(nodo, grafo)

        pos = self.tree_layout(grafo, root="0000")

        # Dibujar el grafo
        plt.figure(figsize=(10, 8))
        nx.draw(grafo, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        plt.title("Grafo del árbol")
        plt.show()

    def get_legal_moves(self, board):
        return list(board.legal_moves)

if __name__ == "__main__":
    board = chess.Board()
    mcts_agent = MCTS("Pythons/Models/NNG/model_200_model_gran.h5", 50)
    best_move = mcts_agent.choose_action(board, [])

    print("Mejor movimiento encontrado:", best_move)

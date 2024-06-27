import pandas as pd
import numpy as np
import chess
from sklearn.model_selection import train_test_split
import csv
import random
import matplotlib.pyplot as plt

class bd_creator:
    def __init__(self) -> None:
        self.inputmeta = None
        self.inputboard = None
        self.data_labels = None
        self.data_test = None

    def boardstate(self, fen):
        try:
            board = chess.Board(str(fen.iloc[0]))
            fstr = str(fen.iloc[0])
        except ValueError as e:
            print("Error: Debes introducir un número válido.")
            return 0

        # Boolean to integer mappings
        WCKI = int(board.has_kingside_castling_rights(chess.WHITE))
        WCQ = int(board.has_queenside_castling_rights(chess.WHITE))
        WCH = int(board.is_check())
        BCKI = int(board.has_kingside_castling_rights(chess.BLACK))
        BCQ = int(board.has_queenside_castling_rights(chess.BLACK))
        BCH = int(board.was_into_check())

        fw = [WCKI, WCQ, WCH]
        fb = [BCKI, BCQ, BCH]

        # Piece to integer mapping
        piece_map = {
            'p': -1, 'n': -3, 'b': -4, 'r': -5, 'q': -9, 'k': -100,
            'P': 1, 'N': 3, 'B': 4, 'R': 5, 'Q': 9, 'K': 100, '.': 0
        }

        # Convert board to a list of integers
        bstr = [piece_map.get(char, 0) for char in str(board).replace('\n', '') if char in piece_map]

        # Flip the board if it's Black's turn
        if "w" not in fstr:
            bstr = [-piece for piece in bstr][::-1]
            fw, fb = fb, fw

        # Combine castling/check info with the board state
        BITBOARD = fw + fb + bstr

        return BITBOARD



    # Funcio que permet transformar una parella de tauler + evaluacio en un valor general
    # en cas que sigui jaque mate l'evalua com a 10000 o -10000 depenen del color que guanyi

    def strfix(self, fen, tr):

        fstr = str(fen)

        if '#' in str(tr):
            if '-' in tr:
                t = -10000
            else:
                t = 10000
        elif '\ufeff+23' in str(tr):
            t = 0
        else:
            t = int(tr)

        if "w" not in fstr:
            t = t*-1
        return t


    def train_data(self, nrows):
        self.seleccionar_lineas_aleatorias("drive/MyDrive/TFG/chessData.csv", "drive/MyDrive/TFG/rand_lines_data.csv", nrows)

        data = pd.read_csv("drive/MyDrive/TFG/rand_lines_data.csv")

        # creem array de labels per la red neruronal
        data_labels = data
        data_labels.columns = ['col1', 'col2']
        data_labels = data_labels.astype(str)
        self.data_labels = data_labels.apply(lambda x: self.strfix(x['col1'], x['col2']), axis=1)

        # creem les dades que permetran entrenar a la red neuronal
        label_columns = [1]
        data_features = data.drop(columns=data.iloc[:, label_columns])
        data_features = data_features.apply(self.boardstate, axis=1)
        data_features = data_features.apply(pd.Series)


        # gardem en un array el input del taulell
        input2_columns = [0, 1, 2, 3, 4, 5]
        inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns])

        self.inputboard = np.array(inputboard)

        # guardem en un array el input de la info del estat
        inputmeta = data_features.iloc[:, input2_columns]

        self.inputmeta = np.array(inputmeta)


    def test_data(self, nrows):

        self.seleccionar_lineas_aleatorias("drive/MyDrive/TFG/chessData.csv", "drive/MyDrive/TFG/rand_lines_data.csv", nrows)

        data = pd.read_csv("drive/MyDrive/TFG/rand_lines_data.csv")

        # creem array de labels per la red neruronal
        data_labels = data
        data_labels.columns = ['col1', 'col2']
        data_labels = data_labels.astype(str)
        self.data_labels = data_labels.apply(lambda x: self.strfix(x['col1'], x['col2']), axis=1)

        # creem les dades que permetran entrenar a la red neuronal
        label_columns = [1]
        data_features = data.drop(columns=data.iloc[:, label_columns])
        data_features = data_features.apply(self.boardstate, axis=1)
        data_features = data_features.apply(pd.Series)


        # gardem en un array el input del taulell
        input2_columns = [0, 1, 2, 3, 4, 5]
        inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns])

        self.inputboard = np.array(inputboard)

        # guardem en un array el input de la info del estat
        inputmeta = data_features.iloc[:, input2_columns]

        self.inputmeta = np.array(inputmeta)

        return (self.inputboard, self.inputmeta, self.data_labels)

    def seleccionar_lineas_aleatorias(self, input_file, output_file, num_lines):
        # Definir los intervalos de evaluación
        num_lines_per_interval = int(num_lines / 4)
        intervals = {
            "[-1000, -500)": [],
            "[-500, 0)": [],
            "[0, 500)": [],
            "[500, 1000]": [],
        }
        extra = []

        # Leer el archivo de entrada
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Leer el encabezado si existe

            for line in reader:
                # Asumimos que la evaluación está en la última columna y es numérica
                try:
                    evaluation = int(line[-1])
                except ValueError:
                    continue  # Si no se puede convertir a float, se omite la línea

                # Clasificar la línea en el intervalo correspondiente
                if -100000 <= evaluation < -1000:
                    intervals["[-1000, -500)"].append(line)
                elif -1000 <= evaluation < 0:
                    intervals["[-500, 0)"].append(line)
                elif 0 <= evaluation < 1000:
                    intervals["[0, 500)"].append(line)
                elif 1000 <= evaluation <= 100000:
                    intervals["[500, 1000]"].append(line)

        # Seleccionar num_lines_per_interval de cada interval
        selected_lines = []
        for interval, lines in intervals.items():
            if len(lines) < num_lines_per_interval:
                print(f"Warning: not enough lines in interval {interval}")
                selected_lines.extend(lines)
            else:
                selected_lines.extend(random.sample(lines, num_lines_per_interval))
        
        
        # Guardar las líneas seleccionadas en el archivo de salida
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Escribir el encabezado
            writer.writerows(selected_lines)

    def train_test(self, nrows, test_size=0.2):

        self.seleccionar_lineas_aleatorias("BD/Clasified/chessData.csv", "BD/Clasified/rand_lines_data.csv", nrows)

        data = pd.read_csv("BD/Clasified/rand_lines_data.csv")
        # creem array de labels per la red neruronal
        data_labels = data
        data_labels.columns = ['col1', 'col2']
        data_labels = data_labels.astype(str)
        data_labels = data_labels.apply(lambda x: self.strfix(x['col1'], x['col2']), axis=1)
        self.data_labels = np.array(data_labels)

        # creem les dades que permetran entrenar a la red neuronal
        label_columns = [1]
        data_features = data.drop(columns=data.iloc[:, label_columns])
        data_features = data_features.apply(self.boardstate, axis=1)
        data_features = data_features.apply(pd.Series)


        # gardem en un array el input del taulell
        input2_columns = [0, 1, 2, 3, 4, 5]
        inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns])

        self.inputboard = np.array(inputboard)

        # guardem en un array el input de la info del estat
        inputmeta = data_features.iloc[:, input2_columns]

        self.inputmeta = np.array(inputmeta)


        im_train, im_test, lab_train, lab_test = train_test_split(self.inputmeta, self.data_labels, test_size=test_size, random_state=123)
        ib_train, ib_test, _, _ = train_test_split(self.inputboard, self.data_labels, test_size=test_size, random_state=123)
        return (im_train, im_test, ib_train, ib_test, lab_train, lab_test)
        
        


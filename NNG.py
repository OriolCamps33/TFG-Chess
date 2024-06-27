import tensorflow as tf
from BD_creator import * 

class NNG_Agent:
    def __init__(self, filepath,
                metric =[tf.keras.metrics.MeanAbsoluteError()],
                opt = tf.keras.optimizers.Adam(),
                los = tf.keras.losses.MeanSquaredError()) -> None:

        self.model = tf.keras.models.Model()
        
        self.metric = metric
        self.opt = opt
        self.los = los
        self.do_model()

        self.load(filepath)

    def do_model(self):
        input1 = tf.keras.layers.Input(shape=(64,))
        shape1 = tf.keras.layers.Reshape(target_shape=(8, 8, 1))(input1)
        conv1 = tf.keras.layers.Conv2D(kernel_size=(3,3), padding="same", activation="relu", filters=64)(shape1)
        bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-05)(conv1)
        flatten1 = tf.keras.layers.Flatten()(bn1)

        input2 = tf.keras.layers.Input(shape=(6,))

        conc = tf.keras.layers.concatenate([flatten1, input2])

        # Dense layers with regularization and dropout
        Denselayer1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(conc)
        Denselayer1 = tf.keras.layers.BatchNormalization()(Denselayer1)
        Denselayer1 = tf.keras.layers.Dropout(0.3)(Denselayer1)

        Denselayer2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(Denselayer1)
        Denselayer2 = tf.keras.layers.BatchNormalization()(Denselayer2)
        Denselayer2 = tf.keras.layers.Dropout(0.3)(Denselayer2)

        Denselayer3 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(Denselayer2)
        Denselayer3 = tf.keras.layers.BatchNormalization()(Denselayer3)
        Denselayer3 = tf.keras.layers.Dropout(0.3)(Denselayer3)

        Denselayer4 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(Denselayer3)
        Denselayer4 = tf.keras.layers.BatchNormalization()(Denselayer4)
        Denselayer4 = tf.keras.layers.Dropout(0.3)(Denselayer4)

        Denselayer5 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(Denselayer4)
        Denselayer5 = tf.keras.layers.BatchNormalization()(Denselayer5)
        Denselayer5 = tf.keras.layers.Dropout(0.3)(Denselayer5)

        Output = tf.keras.layers.Dense(1, activation='linear')(Denselayer3)

        self.model = tf.keras.models.Model(inputs=[input1, input2], outputs=Output)


    def load(self, name):
        self.model.load_weights(name)
        print("Model", name, "loaded.")


    def get_legal_moves(self, board):
        return list(board.legal_moves)
    
    def boardstate(self, fen):
        board = chess.Board(str(fen.iloc[0]))
        fstr = str(fen.iloc[0])

        if board.has_kingside_castling_rights(chess.WHITE):
            WCKI = 1
        else:
            WCKI = 0
        if board.has_queenside_castling_rights(chess.WHITE):
            WCQ = 1
        else:
            WCQ = 0
        if board.is_check():
            WCH = 1
        else:
            WCH = 0
        if board.has_kingside_castling_rights(chess.BLACK):
            BCKI = 1
        else:
            BCKI = 0
        if board.has_queenside_castling_rights(chess.BLACK):
            BCQ = 1
        else:
            BCQ = 0
        if board.was_into_check():
                BCH = 1
        else:
            BCH = 0
        
        fw = [WCKI, WCQ, WCH]
        fb = [BCKI, BCQ, BCH]

        bstr = str(board)
        bstr = bstr.replace("p", "\ -1")
        bstr = bstr.replace("n", "\ -3")
        bstr = bstr.replace("b", "\ -4")
        bstr = bstr.replace("r", "\ -5")
        bstr = bstr.replace("q", "\ -9")
        bstr = bstr.replace("k", "\ -100")
        bstr = bstr.replace("P", "\ 1")
        bstr = bstr.replace("N", "\ 3")
        bstr = bstr.replace("B", "\ 4")
        bstr = bstr.replace("R", "\ 5")
        bstr = bstr.replace("Q", "\ 9")
        bstr = bstr.replace("K", "\ 100")
        bstr = bstr.replace(".", "\ 0")
        bstr = bstr.replace("\ ", ",")
        bstr = bstr.replace("'", " ")
        bstr = bstr.replace("\n", "")
        bstr = bstr.replace(" ", "")
        bstr = bstr[1:]
        bstr = eval(bstr)
        bstr = list(bstr)
        
        if "w" not in fstr:
            for i in range(len(bstr)):
                bstr[i] = bstr[i] * -1
            
            bstr.reverse()
            fs = fb
            fb = fw
            fw = fs
        
        BITBOARD = fw + fb + bstr
        return BITBOARD

    def get_input(self, fen):
        data_features = pd.DataFrame({'FEN': [fen]})

        
		# creem les dades que permetran entrenar a la red neuronal
        data_features = data_features.apply(self.boardstate, axis=1)
        data_features = data_features.apply(pd.Series)

		
		# gardem en un array el input del taulell
        input2_columns = [0, 1, 2, 3, 4, 5]
        inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns])
        
        inputboard = np.array(inputboard)
		
		# guardem en un array el input de la info del estat
        inputmeta = data_features.iloc[:, input2_columns]
        
        inputmeta = np.array(inputmeta)
		
        return [inputboard, inputmeta]

    def choose_action(self, board, moves):
        
        best_move = moves[0]

        if board.turn == chess.WHITE:
            max_punt = -float('inf')
            for move in moves:

                board.push(move)

                fen_board = board.fen()
                inputs = self.get_input(fen=fen_board)
                punt = self.model.predict(inputs, verbose=0)[0,0]
              
                if max_punt < punt:
                    max_punt = punt
                    best_move = move
                board.pop()
        else:
            max_punt = float('inf')
            for move in moves:

                board.push(move)

                fen_board = board.fen()
                inputs = self.get_input(fen=fen_board)
                punt = self.model.predict(inputs, verbose=0)[0,0]
                
                if max_punt > punt:
                    max_punt = punt
                    best_move = move
                board.pop()

        return best_move
    
    def predict(self, board):
        fen_board = board.fen()
        inputs = self.get_input(fen=fen_board)
        punts = self.model.predict(inputs, verbose=0)[0,0]
        return punts

if __name__ == "__main__":
    agent = NNG_Agent("Models/NNG/model_100.keras")


    




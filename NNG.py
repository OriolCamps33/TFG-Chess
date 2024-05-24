import tensorflow as tf
from BD_creator import * 
import pydotplus
from IPython.display import Image

class NNG_Agent:
    def __init__(self,
                metric =[tf.keras.metrics.MeanAbsoluteError()],
                opt = tf.keras.optimizers.Adam(),
                los = tf.keras.losses.MeanSquaredError()) -> None:

        self.data = bd_creator()

        self.model = tf.keras.models.Model()
        self.metric = metric
        self.opt = opt
        self.los = los

    def do_model(self):

        input1 = tf.keras.layers.Input(shape=(64,))
        shape1 = tf.keras.layers.Reshape(target_shape=(8, 8, 1))(input1)
        conv1 = tf.keras.layers.Conv2D(kernel_size=(8,8), padding="same", activation="relu", filters=64, input_shape=(8,8,1))(shape1)
        bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-05)(conv1)
        conv2 = tf.keras.layers.Conv2D(kernel_size=(8,8), padding="same", activation="relu", filters=64, input_shape=(8,8,1))(bn1)
        bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-05)(conv2)
        flatten1 = tf.keras.layers.Flatten()(bn2)
        input2 = tf.keras.layers.Input(shape=(6,))

        conc = tf.keras.layers.concatenate([flatten1,input2])

        Denselayer1 = tf.keras.layers.Dense(1024, activation='relu')(conc)
        Denselayer2 = tf.keras.layers.Dense(512, activation='relu')(Denselayer1)
        Denselayer3 = tf.keras.layers.Dense(256, activation='relu')(Denselayer2)
        Denselayer4 = tf.keras.layers.Dense(256, activation='relu')(Denselayer3)
        Output = tf.keras.layers.Dense(1, activation='linear')(Denselayer4)


        data_model = tf.keras.models.Model(inputs=[input1, input2], outputs=Output)

        self.model = data_model

    def fit(self, epoch, batch_size=8192):
        self.data.train_data(100000)

        self.model.compile(optimizer=self.opt, loss=self.los,  metrics=self.metric)
        self.model.summary()
        self.model.fit([self.data.inputboard, self.data.inputmeta], self.data.data_labels, epochs=epoch, batch_size=batch_size, shuffle=True)

    def save(self, name):
        self.model.save(name)
        print("Model saved in: ", name)
        
    def load(self, name):
        self.model = tf.keras.models.load_model(name)
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

        max_punt = 0
        best_move = moves[0]
        for move in moves:

            board.push(move)

            fen_board = board.fen()
            inputs = self.get_input(fen=fen_board)
            punt = self.model.predict(inputs, verbose=0)[0,0]

            
            if max_punt < punt:
                max_punt = punt
                best_move = move
            board.pop()

        return best_move

    def print_model(self, dst):

        dot = tf.keras.utils.model_to_dot(self.model, show_shapes=True, show_layer_names=True)
        graph = pydotplus.graph_from_dot_data(dot.to_string())
        Image(graph.create_png())

        #tf.keras.utils.plot_model(self.model, to_file=dst, show_shapes=True, show_layer_names=True)

if __name__ == "__main__":
    import os
    #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # Cambia la ruta según la ubicación de tu instalación de Graphviz
    print(os.pathsep)
    agent = NNG_Agent()
    agent.load("Models/NNG/model_600.keras")

    agent.print_model('Imagenes/Modelos/model1.png')
    print("end")
    




import pandas as pd
import numpy as np
import chess

class bd_creator:

	def __init__(self) -> None:
		self.inputmeta = None
		self.inputboard = None
		self.data_labels = None
		self.test_data = None


	# Funcio que et permet interpretar el taulell, escrit en FEN, 
	# i convertirlo a un estat que et permet ser utilitzat per entrenar la red neuronal
	def boardstate(self, fen):
		board = chess.Board(str(fen.iloc[0]))
		fstr = str(fen.iloc[0])

		# Es guarda informacio del estat concret, com els possibles castles i si hi ha jaque
		if board.has_kingside_castling_rights(chess.WHITE) == True:
			WCKI = 1
		else:
			WCKI = 0
		if board.has_queenside_castling_rights(chess.WHITE) == True:
			WCQ = 1
		else:
			WCQ = 0
		if board.is_check() == True:
			WCH = 1
		else:
			WCH = 0

		if board.has_kingside_castling_rights(chess.BLACK) == True:
			BCKI = 1
		else:
			BCKI = 0
		if board.has_queenside_castling_rights(chess.BLACK) == True:
			BCQ = 1
		else:
			BCQ = 0
		if board.was_into_check() == True:
			BCH = 1
		else:
			BCH = 0

		fw = [WCKI, WCQ, WCH]
		fb = [BCKI, BCQ, BCH]

		# Es transforma el taulell a un array de ints.
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
		# es retorna un array amb la info del estat y el taulell
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

		t = t/10

		return t


	def train_data(self, nrows):
		data = pd.read_csv('BD/Clasified/chessData.csv', nrows=nrows)

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
		
				
		
if __name__ == "__main__":
	creator = bd_creator()
	creator.train_data(100000)
	
        
        


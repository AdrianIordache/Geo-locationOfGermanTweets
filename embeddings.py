from utils import *

def create_embeddings_use(data: pd.DataFrame, column: str = "text", path: str = None, typs: str = "train"):
	assert column in data.columns.tolist(), "[embeddings.py] -> [create_embeddings_use] -> Input column not in dataframe columns"
	embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")	

	if path == None:
		path = typs + "_embeddings.csv"

	text = np.expand_dims(data[column].values, axis = 1)

	tic = time.time()

	embeddings = np.zeros((len(text), 512))

	for idx in range(embeddings.shape[0]):
		embedding_tensor = embed(text[idx])
		sample = np.array(tf.keras.backend.eval(embedding_tensor))
		embeddings[idx] = sample


	csv = pd.DataFrame(embeddings)
	csv.to_csv(path, index = False)

	toc = time.time()

	print("[create_embeddings_use] -> time {}'s".format(toc - tic))


def create_embedding_scapy(data: pd.DataFrame, column: str = "text", path: str = None, typs: str = "train"):
	assert column in data.columns.tolist(), "[embeddings.py] -> [create_embedding_scapy] -> Input column not in dataframe columns"

	tic = time.time()

	embeddings = [nlp(text).vector for text in data[column].values]
	embeddings = np.array(embeddings)

	csv = pd.DataFrame(embeddings)
	csv.to_csv(path, index = False)

	toc = time.time()

	print("[create_embeddings_scapy] -> time {}'s".format(toc - tic))


def create_embeddings_flair(data: pd.DataFrame, column: str = "text", path: str = None, embeddings_type: str = "tranformer", typs: str = "train"):
	assert column in data.columns.tolist(), "[embeddings.py] -> [create_embedding_flair] -> Input column not in dataframe columns"
	assert embeddings_type in ["tranformer", "stacked"]

	import flair
	from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
	from flair.data import Sentence

	flair.device = torch.device('cpu') 

	fast_text_embedding      = WordEmbeddings('de')
	flair_embedding_forward  = FlairEmbeddings('de-forward')
	flair_embedding_backward = FlairEmbeddings('de-backward')

	stacked_embeddings = DocumentPoolEmbeddings([fast_text_embedding, flair_embedding_forward, flair_embedding_backward])

	transformer_embedding = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune = False)

	tic = time.time()

	embeddings = []

	for i, text in enumerate(data[column].values):
		print("sentence {}/{}".format(i, len(data)))
		sentence = Sentence(text)

		if embeddings_type == "stacked":
			stacked_embeddings.embed(sentence)
		elif embeddings_type == "tranformer":
			transformer_embedding.embed(sentence)

		embedding = sentence.embedding.detach().cpu().numpy()
		embeddings.append(embedding)
		
	embeddings = np.array(embeddings)

	columns = ["embedding_{}".format(feature) for feature in range(embeddings.shape[1])]

	csv = pd.DataFrame(embeddings, columns = columns)
	csv.to_csv(path + embeddings_type + "_" + typs + ".csv", index = False)

	toc = time.time()

	print("[create_embeddings_flair] -> [embeddings_type: {}, typs: {}] -> time {}'s".format(embeddings_type, typs, toc - tic))

if __name__ == "__main__":

	# create_embeddings_flair(preprocessed_train, column = 'final_text', path = "", embeddings_type = "stacked", typs = "train")
	# create_embeddings_flair(preprocessed_valid, column = 'final_text', path = "", embeddings_type = "stacked", typs = "valid")
	# create_embeddings_flair(preprocessed_test,  column = 'final_text', path = "", embeddings_type = "stacked", typs = "test")

	create_embeddings_flair(preprocessed_train, column = 'final_text', path = "", embeddings_type = "tranformer", typs = "train")
	create_embeddings_flair(preprocessed_valid, column = 'final_text', path = "", embeddings_type = "tranformer", typs = "valid")
	create_embeddings_flair(preprocessed_test,  column = 'final_text', path = "", embeddings_type = "tranformer", typs = "test")
from utils import *

import translators as ts
from googletrans import Translator

translator = Translator()

def translate(text: str):
	def translate_(text: str):
		assert isinstance(text, str), "[Translator.py] -> [translate.py] -> The input is not a string"
		print("Initial Sentence: " + str(text)) 

		try:
			result = ts.yandex(text, to_language = "de")
			print("Translated Sentence: " + str(result)) 
			return (1, result)
		except:
			print("If you want try again, sometimes it works")
			print("Thanks Google")
			return (0, None)

	exit_code = 0
	while exit_code == 0:
		(exit_code, translated) = translate_(text)

	return translated

def translate_dataframe(data: pd.DataFrame, savefile: str, verbose: bool = False):
	def translate_dataframe_(data: pd.DataFrame, verbose: bool = False, counter: int = 0):
		assert isinstance(data, pd.DataFrame), "[Translator.py] -> [translate_dataframe.py] -> The input is not a pd.Dataframe"
		assert "text" in data.columns.tolist(), "[Translator.py] -> [translate_dataframe.py] -> 'Text' column missing from dataframe"

		text = data["text"].iloc[counter]
				
		try:
			result = ts.google(text)
			print("Works... Sentence {}/{} Translated".format(counter, data.shape[0]))
			counter += 1
			return (1, counter, result)
		except:
			print("If you want try again, sometimes it works")
			print("Thanks Google")
			return (0, counter, None)

	data["translated"] = ""
	counter = 0 
	while counter < data.shape[0]:
		exit_code = 0
		while exit_code == 0:
			(exit_code, counter, translated_text) = translate_dataframe_(data, verbose, counter)

		data["translated"].iloc[counter - 1] = translated_text

	data.to_csv(savefile, index = False)



def translate_text_by_index(data: pd.DataFrame, idx: int):
	def translate_text_by_index_(data: pd.DataFrame, idx: int):
		assert "text" in data.columns.tolist(), "[Translator.py] -> [translate_text_by_index.py] -> 'Text' column missing from dataframe"
		text = data[data["id"] == idx].text.values[0]
		print("Initial Sentence: " + str(text)) 
		try:
			result = translator.translate(text)
			print("Translated Sentence: " + str(result.text)) 
			return 1
		except:
			print("If you want try again, sometimes it works")
			print("Thanks Google")
			return 0

	exit_code = 0
	while exit_code == 0:
		exit_code = translate_text_by_index_(data, idx)

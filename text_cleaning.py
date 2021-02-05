from utils import *
from embeddings import *
from translator import *



class TextCleaner:
    def __init__(self, data: pd.DataFrame, lemmatize: bool = True):
        self.data = data
        self.stop = set(stopwords.words('german'))
        self.lemmatize = lemmatize
        
        self.clean()

    def clean(self):
        aux_columns = ["text_clean", "tokenized", "lower", "stopwords_removed"]

        self.data['text_clean'] = self.data['text'].apply(lambda x: TextCleaner.remove_URL(x))
        self.data['text_clean'] = self.data['text_clean'].apply(lambda x: TextCleaner.remove_emoji(x))
        self.data['text_clean'] = self.data['text_clean'].apply(lambda x: TextCleaner.remove_html(x))
        self.data['text_clean'] = self.data['text_clean'].apply(lambda x: TextCleaner.remove_punct(x))

        self.data['tokenized'] = self.data['text_clean'].apply(lambda x: [token.text for token in nlp(x)])

        self.data['lower'] = self.data['tokenized'].apply(lambda x: [word.lower() for word in x])

        self.data['stopwords_removed'] = self.data['lower'].apply(lambda x: [word for word in x if word not in self.stop])

        self.data['final_text'] = [' '.join(map(str, l)) for l in self.data['stopwords_removed']]

        if self.lemmatize:
            self.data['lemmatized'] = self.data['final_text'].apply(lambda x: [token.lemma_ for token in nlp(x)])
            self.data['final_text'] = [' '.join(map(str, l)) for l in self.data['lemmatized']]
            aux_columns.append('lemmatized')


        self.data['final_text'] = self.data['final_text'].apply(lambda x: ' '.join(x.split()))

        self.data.drop(aux_columns, axis = 1, inplace = True)


    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)


    def remove_html(text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)


    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)


if __name__ == "__main__":
    train_df = pd.read_csv(PATH_TO_TRAIN, sep = ',', header = None)
    train_df.columns = COLUMNS

    valid_df = pd.read_csv(PATH_TO_VALID, sep = ',', header = None)
    valid_df.columns = COLUMNS

    test_df = pd.read_csv(PATH_TO_TEST, sep = ',', header = None)
    test_df.columns = TEST_COLUMNS

    train_df = TextCleaner(train_df).data
    valid_df = TextCleaner(valid_df).data
    test_df  = TextCleaner(test_df).data

    display(train_df.head(n = 10))

    # create_embeddings_use(train_df, column = "final_text", typs = "train", path = "data/embeddings/version-4/train_embeddings.csv")
    # create_embeddings_use(valid_df, column = "final_text", typs = "valid", path = "data/embeddings/version-4/valid_embeddings.csv")
    # create_embeddings_use(test_df,  column = "final_text", typs = "test", path = "data/embeddings/version-4/test_embeddings.csv")




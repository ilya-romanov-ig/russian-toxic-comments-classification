from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('russian'))
stop_words = stop_words.union(set([
    'это', 
    'очень', 
    'вообще', 
    'всё', 
    'ещё', 
    "просто", 
    "почему", 
    "которые",
    "который",
    "просто",
    "это",
    "пока",
    "хотя",
    "вроде",
    "тебе",
    "твой",
    "чтото",
    "такой",
    "такие",
    "такое",
]))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language='russian')
    cleaned_tokens = []
    for t in tokens:
        if t not in stop_words and t.isalpha() and len(t) > 2:
            lemma = lemmatizer.lemmatize(t)
            cleaned_tokens.append(lemma)
    return cleaned_tokens

def extract_tokens(df, istoxic=0.0):
    tokens = []
    for t in df[df['toxic'] == istoxic]['comment']:
        tokens.extend(preprocess_text(t))
    return tokens
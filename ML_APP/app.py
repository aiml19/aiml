from flask import Flask,render_template,url_for,request
# # Let us first start with the date to note down timing
# import warnings
# warnings.filterwarnings('always')

from datetime import datetime
def time_now():
    return datetime.now().strftime("%H:%M:%S")

def print_msg(*msg):
    print(time_now(),":",*msg)

## Import all the required modules
print_msg("importing modules")

## most used packages
import re,json

### NLP packages
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer

## Keras packages
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  load_model

# sklearn packages
from sklearn import preprocessing


    
print_msg("completed importing modules")

print_msg("loading functions")
          
## Functions to loading data, build models and data preprocessing
## New Log table to capture metrics from different model iterations 
    

# Data preprocessing 
# When we set the flag deacc=True , the function removes punctuations also 
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True , min_len=3, max_len=20))  # deacc=True removes punctuations

# covert list to sentences
def words_to_sent(sentences):
    final=[]
    for sentence in sentences:
        local=" ".join(sentence)
        final.append(local)
    return final

## Stopword ,duplicate word removal
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'received'])

def remove_stopwords_duplicate(texts):
    final=[]
    for doc in texts:
        local=[]
        for word in simple_preprocess(str(doc)):
            if word not in local:
                local.append(word)
        final.append(local)
    return final

## find POS for each word for lemmatizer
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,  "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_document(documents):
    ## Convert to lower case
    documents = [sent.lower() for sent in documents]
    # Remove Emails
    documents = [re.sub('\S*@\S*\s?', ' ', sent) for sent in documents]
    # Remove new line characters
    documents = [re.sub('\s+', ' ', sent) for sent in documents]
    # Remove _
    documents = [re.sub('_', ' ', sent) for sent in documents]
    # Remove Numbers
    documents = [re.sub('\d+', ' ', sent) for sent in documents]
    # Remove  distracting single quotes
    documents = [re.sub('\'', ' ', sent) for sent in documents]
    # Remove all non word characters
    documents = [re.sub('\W', ' ', sent) for sent in documents]
    
    document_words = list(sent_to_words(documents))
    document_words = remove_stopwords_duplicate(document_words)
    
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
#     print_msg("lemmatization started")
    hl_lemmatized = []
    for tokens in document_words:
        lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
        hl_lemmatized.append(lemm)
    document_words=hl_lemmatized   
#     print_msg("lemmatization ended")
    
    return document_words

def model_load():
    with open('tokenizer.json') as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    model = load_model('my_model.h5')
    
    groups = []
    with open('group.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            groups.append(currentPlace)
        
    le = preprocessing.LabelEncoder()
    le.fit(groups)
    
    return le,tokenizer,model

def model_predict(text):
    document=[text]
    documents=preprocess_document(document)
    sequence = tokenizer.texts_to_sequences(documents)
    maxlen=model.input.shape[1]
    X = pad_sequences(sequence, maxlen = maxlen, padding='post',truncating='post')
    y=model.predict_classes(X)
    return "".join(le.inverse_transform(y))
print_msg("Imported modules and new functions completed")

le,tokenizer,model=model_load()



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		comment = request.form['comment']
		my_prediction=model_predict(comment)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
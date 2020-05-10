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
import numpy as np
import pandas as pd
import re,os,io,json

### NLP packages
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer

## Keras packages
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input ,Flatten,BatchNormalization
from tensorflow.keras.layers import Embedding,GlobalMaxPool1D,Bidirectional,SpatialDropout1D
from tensorflow.keras.models import Sequential,load_model

# sklearn packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier   
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn import metrics,preprocessing,svm
from sklearn.model_selection import train_test_split

    
print_msg("completed importing modules")

print_msg("loading functions")
          
## Functions to loading data, build models and data preprocessing
## New Log table to capture metrics from different model iterations 
try:                              
    len(log)
except:
    print("create log table")
    log_cols = ["groups","model_name","model_column","data_set", "Accuracy","Precision Score","Recall Score","F1-Score","kappa_score"]
    log = pd.DataFrame(columns=log_cols)

# Function to capture metrics in the log table, save table to log.xlsx file with every run
def metric_update(y_test,y_pred):
    global model_column,model_name,data_set
    global log,itr_cnt
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,average='macro',labels=np.unique(y_pred))
    recall = metrics.recall_score(y_test,y_pred,average='macro',labels=np.unique(y_pred))
    f1_score = metrics.f1_score(y_test,y_pred,average='macro',labels=np.unique(y_pred))
    kappa_score=metrics.cohen_kappa_score(y_test,y_pred)
    col_data=[msg_grp,model_name,model_column,data_set,accuracy,precision,recall,f1_score,kappa_score]
    log_entry = pd.DataFrame([col_data], columns=log_cols)
    log = log.append(log_entry)
    itr_cnt=itr_cnt+1
    print_msg("completed iteration ={}".format(itr_cnt))
    print_msg(metrics.classification_report(y_test,y_pred))


# Function to load data and perform preprocessing,add new columns in dataframe    
def load_data(file,prefix):        
    print_msg("loading file=",file)
    df = pd.read_excel(file)
    df= df.drop("Caller" , axis=1)
    df.columns=["short_description","long_description","assigned_group"]
    df["combined_description"]=df["short_description"]+" "+df["long_description"]
    df.dropna(inplace=True) ## Not many null ,so can safely drop the rows
    df,name=preprocess_column(df,"short_description")
    df,name=preprocess_column(df,"combined_description")
    df["assigned_group_org"]=df["assigned_group"]
    df.dropna(subset=['combined_description_text'],inplace=True)
    df.short_description_text[df.short_description_text.isnull()]=df.combined_description_text[df.short_description_text.isnull()]
    df.to_excel(prefix+file)
    return df
    

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

def preprocess_column(df,column_name):
    new_column=column_name+"_list"
    new_column1=column_name+"_text"
    try:
        df[new_column].shape
        print_msg("pre-processing was already done")
    except:
        print_msg("process started for "+column_name)
        documents = df[column_name].values.tolist()
        document_words=preprocess_document(documents)
        df[new_column]=document_words
        df[new_column1]=words_to_sent(document_words)
        print_msg("process finished for "+column_name)
    return df,new_column

## sklearn models with default parameters
def sklearn_model(column_name):
    global model_column
    global model_name
    global data_set
    print_msg("fitting all classic models with column_name=",column_name)
    model_column=column_name
    
    X = df[model_column]
    y = df.assigned_group.astype('category')
 
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(X)
    
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_bow)

    model = LogisticRegression()
    model_name = "LogisticRegression"
    sklearn_model_fit(model,X_bow,X_tfidf,y)

    model = DecisionTreeClassifier(criterion = 'entropy' )
    model_name ="DecisionTreeClassifier"
    sklearn_model_fit(model,X_bow,X_tfidf,y)

    model= RandomForestClassifier()
    model_name = "RandomForestClassifier"
    sklearn_model_fit(model,X_bow,X_tfidf,y)

    model = AdaBoostClassifier(n_estimators= 20)
    model_name = "AdaBoostClassifier"
    sklearn_model_fit(model,X_bow,X_tfidf,y)

    model = BaggingClassifier()
    model_name = "BaggingClassifier"
    sklearn_model_fit(model,X_bow,X_tfidf,y)
   
#     model = GradientBoostingClassifier()
#     name = "GradientBoostingClassifier"
#     classic_model_fit(name,model,X_bow,X_tfidf,y)

    model = GaussianNB()
    X_array_bow  = X_bow.toarray()
    X_array_tfidf  = X_tfidf.toarray()
    model_name = "GaussianNB"
    sklearn_model_fit(model,X_array_bow,X_array_tfidf,y)

    model = svm.SVC()
    model_name = "svm.svc"
    sklearn_model_fit(model,X_bow,X_tfidf,y)
    
def sklearn_model_fit(model,X_bow,X_tfidf,y):
    global model_column,model_name,data_set
    data_set="Bow"
    X_train, X_test, y_train, y_test = train_test_split(X_bow,y, test_size=0.10, random_state=42,stratify=y)
    print_msg("working on",msg_grp,model_column,model_name,data_set)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    metric_update(y_test,y_pred)
    
    data_set="TFIDF"
    print_msg("working on",msg_grp,model_column,model_name,data_set)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y, test_size=0.10, random_state=42,stratify=y)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    metric_update(y_test,y_pred) 
    
try:
    len(embeddings)
except:
    embeddings = {}
    
def embedding(glove_file):
    global embeddings
    
    if len(embeddings) < 1000:
        print_msg("embedding length",len(embeddings))
        print_msg("Building Embeddings from "+glove_file)

        for o in open(glove_file,encoding="utf8"):
            word = o.split(" ")[0]
            # print_msg(word)
            embd = o.split(" ")[1:]
            embd = np.asarray(embd, dtype='float32')
            # print_msg(embd)
            embeddings[word] = embd
    else:
        print_msg("Embeddings from ",glove_file,"already exists")
    print_msg("No. of embeddings = ", len(embeddings))  

def keras_model(column_name,epoch=5,fit=0):
    global model_column,model_name,data_set
    model_column=column_name
    model_name="Keras LSTM Model"
    data_set="processed"
    
    print_msg("working on",msg_grp,model_column,model_name,data_set)
    
    max_features = 10000
    #epoch = 20
    batch_size = 100
    
    documents = df[model_column].values.tolist()
    max_allowed=max(df["short_description_text"].apply(lambda x: len(x.split(" "))))*2
    max_allowed=40
    maxlen = max(df[model_column].apply(lambda x: len(x)))
    #print_msg("maxlen before=",maxlen) 
    if maxlen>max_allowed:
        maxlen=max_allowed
    print_msg("maxlen after=",maxlen)    

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(documents))
    sequence = tokenizer.texts_to_sequences(documents)
    tokenizer_json = tokenizer.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        
    word_index = tokenizer.word_index
    vocab_size = len(word_index)+1
    
    print_msg("My vocabulary size = ",vocab_size)
    X = pad_sequences(sequence, maxlen = maxlen, padding='post',truncating='post') 
    
    groups = list(df.assigned_group.unique())
    with open('group.txt', 'w') as filehandle:
        for listitem in groups:
            filehandle.write('%s\n' % listitem)
        
    le = preprocessing.LabelEncoder()
    le.fit(groups)
    y=to_categorical(le.transform(df.assigned_group))   

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state = 42,stratify=y)   
    
    glove_file='glove.6B.200d.txt'
    embedding(glove_file)
    embedding_size=embeddings['the'].shape[0]
    
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, weights = [embedding_matrix],input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(df.assigned_group.nunique()*2, return_sequences = True,recurrent_dropout=0.1, dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(df.assigned_group.nunique()*2, activation="relu"))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(df.assigned_group.nunique(), activation="softmax"))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['acc'])
    print_msg(model.summary())
    
    if fit>0:
        print_msg('running model.fit')
        history = model.fit(X_train, y_train, epochs = epoch, batch_size=batch_size, validation_split=0.05,shuffle= True,verbose = 1)
        model.save('my_model.h5')
        y_predict=model.predict_classes(X_test)
        y_test1=np.argmax(y_test,axis=1)
        
        metric_update(y_test1, y_predict)
        
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
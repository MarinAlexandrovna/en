import streamlit as st

import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
import pandas as pd
import nltk
import spacy
import pysrt
import os
import re
 
trained_model = pickle.load(open(model, 'rb'))


st.title("Определение уровня сложности субтитров")
st.markdown("---")
uploaded_file = st.file_uploader("Загрузите файл", type = "srt", key=None)

def text_sub(path):
    films = os.listdir(path = path)
    data = {}
    for i in range(len(films)):
        subs = pysrt.open(path + films[i], encoding='iso-8859-1')
        subt=[]
        for sub in subs:
            subt.append(sub.text_without_tags.split('\n'))
        data[films[i]] = subt
    data = pd.DataFrame({'Movie': films, 'Subtitles': data.values()})
    return data

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


if uploaded_file is not None:
	data_1 = text_sub(uploaded_file)
	data_2['Subtitles'] =data_1['Subtitles'].map(lambda s:preprocess(s)) 

	x = data_2['Subtitles']
	result = trained_model.predict(x)

	st.write(f"Your prediction is: {result}")





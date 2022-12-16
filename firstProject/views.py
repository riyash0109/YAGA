#i created this file

from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    params = {'name':'Yash' , 'place':'PG'}
    #return HttpResponse("<h1>hello world</h1>")
    return render(request, 'index.html' , params)

def about(request):
    return HttpResponse("This is created by Yash")

def form(request):
    #we got the text
    print(request.GET.get('text1' , 'default'))
    dtext = request.GET.get('text1' , 'default')
    params = {'text1': dtext}
    return render(request, 'getText.html' , params)


#########################################################################################################################

def search2(request):
    from collections import defaultdict
    slots = defaultdict(list)
    h = 9
    m = 0
    # slots = {}
    for i in range(14):
        if(h == 13):
            h+=1

    print("SLOT ",i+1," at ",h,":",m)
    slots[i].append(h)
    slots[i].append(m)
    m += 30
    if(m==60):
        m = 0 
        h += 1

    # print(slots)
    # slots are represented in manner daynumber_slotnumber. Each slot will be multiplied by 12(for each quater)
    st = ["chandra,sentilnathan,rohini","chandra,rohini,sagaya","chandra,sagaya,sentilnathan"]
    tch = defaultdict(list)
    n = 1
    # while(n):
    #   t = input("Teacher code : ")
    #   s = input("Teacher slots seperated by commas")
    #   tch[t].append(s)
    #   n = int(input("0 : Exit\n1 : To continue "))
    tch = {'chandra': ['112,115,118,313'], 'sentilnathan': ['112,116,412,118'], 'sagaya': ['116,313,513,118'], 'rohini': ['112,,313,116,514']}
    # print(tch)
    # c = [11,12,53,54,56]
    # d = [12,13,55,47,54]
    # e = [11,12,53,36,28,43,54]

    # for i in tch['c']:
    #   print(i)
    st = ["chandra,sentilnathan,rohini","chandra,rohini,sagaya","chandra,sagaya,sentilnathan"]
    # print(st)
    nm = ["Yash","Akshay","Gyan"]
    j = 0
    pt = []
    for i in st:
    
        l = i.split(",")
        # print("s",l)
        t1 = []
        t2 = []
        t3 = []
        for i in tch[l[0]]:
            t1 = i.split(",")
        for i in tch[l[1]]:
            t2 = i.split(",")
        for i in tch[l[2]]:
            t3 = i.split(",")
            # print(t1)
        # print(t1)
        for i in t1:
            # print(tch[l[1]])
            if (i in t2 and i in t2):
                # print("For ",nm[j]," Available slots are : ",i)
                s = "For "+nm[j]+" Available slots are : "+i
                pt.append(s)
        j +=1 

    return render(request, 'new_results.html', {"result" : pt})



###################################################search 1 function#####################################################



def search1(request):

    s2 = request.GET["s2"]
    import pandas as pd
    df = pd.read_csv("firstProject\interest.csv")
    x = s2
    x =  x.upper()
    try:
        y = df.nlargest(2, x)
        return render(request, 'result.html', {"result" : y['NAME'].to_string(index=False)})
    except:
        return render(request, 'result.html', {"result" : "!!!INVALID INPUT!!!"})


###################################################################################################################################3

def search(request):
    s1 = request.GET["s1"]

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import SnowballStemmer
    from rake_nltk import Rake

    import re
    import string
    import requests
    import numpy as np
    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from nltk.corpus import stopwords

    nltk.download('all')

    df = pd.read_csv('firstProject\profile.csv')
    # df.head()

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer 

    count = CountVectorizer()
    count_matrix = count.fit_transform(df['keywords'])

    #create a Series for movie titles so they are associated to an ordered numerical list, we will use this later to match index
    indices = pd.Series(df.index)
    indices[:5]

    #Shape count_matrix
    count_matrix

    #Convert sparse count_matrix to dense vector
    c = count_matrix.todense()
    # c

    # print(count_matrix[0,:])
    count.vocabulary_

    # generating the cosine similarity matrix

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # cosine_sim


    #Lets build a function that takes in movie and recommends top n movies

    def recommendations(title,n,cosine_sim = cosine_sim):
        recommended_movies = []
        
        #get index of the movie that matches the title
        idx = indices[indices == title].index[0]
        
        #find highest cosine_sim this title shares with other titles extracted earlier and save it in a Series
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        
        #get indexes of the 'n' most similar movies
        top_n_indexes = list(score_series.iloc[1:n+1].index)
        # print(top_n_indexes)
        
        #populating the list with titles of n matching movie
        for i in top_n_indexes:
            recommended_movies.append(list(df.index)[i])
            
        return recommended_movies


    import spacy

    from spacy.lang.en.stop_words import STOP_WORDS
    spacy_nlp = spacy.load('en_core_web_sm')

    #create list of punctuations and stopwords
    punctuations = string.punctuation
    stop_words = spacy.lang.en.stop_words.STOP_WORDS


    #function for data cleaning and processing
    #This can be further enhanced by adding / removing reg-exps as desired.

    def spacy_tokenizer(sentence):
    
        #remove distracting single quotes
        sentence = re.sub('\'','',sentence)

        #remove digits adnd words containing digits
        sentence = re.sub('\w*\d\w*','',sentence)

        #replace extra spaces with single space
        sentence = re.sub(' +',' ',sentence)

        #remove unwanted lines starting from special charcters
        sentence = re.sub(r'\n: \'\'.*','',sentence)
        sentence = re.sub(r'\n!.*','',sentence)
        sentence = re.sub(r'^:\'\'.*','',sentence)
        
        #remove non-breaking new line characters
        sentence = re.sub(r'\n',' ',sentence)
        
        #remove punctunations
        sentence = re.sub(r'[^\w\s]',' ',sentence)
        
        #creating token object
        tokens = spacy_nlp(sentence)
        
        #lower, strip and lemmatize
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        
        #remove stopwords, and exclude words less than 2 characters
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
        
        #return tokens
        return tokens


    # print ('Cleaning and Tokenizing...')
    df['new keywords'] = df['PROFILE'].map(lambda x: spacy_tokenizer(x))


    df_article = df['new keywords']

    # from wordcloud import WordCloud
    # import matplotlib.pyplot as plt

    # series = pd.Series(np.concatenate(df_article)).value_counts()[:100]
    # wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)

    # plt.figure(figsize=(15,15), facecolor = None)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()


    from gensim import corpora

    #creating term dictionary
    dictionary = corpora.Dictionary(df_article)

    #filter out terms which occurs in less than 4 documents and more than 20% of the documents.
    #NOTE: Since we have smaller dataset, we will keep this commented for now.

    #dictionary.filter_extremes(no_below=4, no_above=0.2)

    #list of few which which can be further removed
    stoplist = set('hello and if this can would should could tell ask stop come go')
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)


    #print top 50 items from the dictionary with their unique token-id
    dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]


    corpus = [dictionary.doc2bow(desc) for desc in df_article]

    word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]



    import gensim
    article_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    article_lsi_model = gensim.models.LsiModel(article_tfidf_model[corpus], id2word=dictionary, num_topics=300)


    gensim.corpora.MmCorpus.serialize('article_tfidf_model_mm', article_tfidf_model[corpus])
    gensim.corpora.MmCorpus.serialize('article_lsi_model_mm',article_lsi_model[article_tfidf_model[corpus]])



    #Load the indexed corpus
    article_tfidf_corpus = gensim.corpora.MmCorpus('article_tfidf_model_mm')
    article_lsi_corpus = gensim.corpora.MmCorpus('article_lsi_model_mm')

    from gensim.similarities import MatrixSimilarity

    article_index = MatrixSimilarity(article_lsi_corpus, num_features = article_lsi_corpus.num_terms)
    article_index


    from operator import itemgetter

    from operator import itemgetter

    def search_faculty(search_term):

        query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
        query_tfidf = article_tfidf_model[query_bow]
        query_lsi = article_lsi_model[query_tfidf]

        article_index.num_best = 1

        article_list = article_index[query_lsi]

        article_list.sort(key=itemgetter(1), reverse=True)
        faculty_detail = []

        # print(article_list)

        for j, movie in enumerate(article_list):

            faculty_detail.append (
                {
                    #'Relevance': round((movie[1] * 100), 2),
                    df['NAME'][movie[0]],
                    #'Title of Article': df['Title of Article'][movie[0]]
                }

            )
            if j == (article_index.num_best-1):
                break

        return faculty_detail





    # search for faculty who are related to below search parameters
    # inn = input("Enter the string : ")
    ab = search_faculty(s1)


    for i in ab[0]:
        aas = i


    return render(request, 'result.html', {"result" : aas})
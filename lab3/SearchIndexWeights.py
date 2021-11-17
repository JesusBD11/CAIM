"""
.. module:: SearchIndexWeight

SearchIndex
*************

:Description: SearchIndexWeight

    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match

    --nhits changes the number of documents to retrieve

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 10:56 

"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q



from elasticsearch.client import CatClient

import numpy as np

__author__ = 'bejar'

#PARAMETROS#
alpha = 0.8
beta = 0.2
nrounds = 5
k = 100
R = 4
###########

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = {}

    for (t, w),(_, df) in zip(file_tv, file_df):
        #
        # Something happens here
        #
        tf = w / max_freq
        idf = dcount / df
        idf = np.log(idf)
        res = tf * idf

        tfidfw[t] = res
        #tfidfw.append((t, res))

    return normalize(tfidfw)


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    #
    # Program something here
    #
    sq = sum(tw.values())
    sr = np.sqrt(sq)
    result = {l: v/sr for l,v in tw.items()}
    return result
    

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


def getDic(query):
    """
    Convierte una lista en diccionario
    """
    dic = {}
    for i in query:
        if '^' in i:
            word, value = i.split('^')
            value = float(value)
        else: 
            word = i
            value = 1.0

        dic[word] = value

    return normalize(dic)


def toQuery(dic):
    """
    Convierte un diccionario en una query
    """
    query = []
    for i in dic:
        word = i + '^' + str(dic[i])
        query.append(word)
    return query


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nhits', default=10, type=int, help='Number of hits to return')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    query = args.query
    print(query)
    nhits = args.nhits

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)
       

        if query is not None:
            #Ejecutamo nround veces la regla de rocchio
            for i in range(1,nrounds):
                q = Q('query_string',query=query[0])
                for i in range(1, len(query)):
                    q &= Q('query_string',query=query[i])

                s = s.query(q)
                #k documentos mas relevantes
                response = s[0:k].execute()
                #nhits documentos a mostrar
                mostrar = s[0:nhits].execute()

                #Pasamos query a diccionario
                dic = getDic(query)
                #Diccionario donde acumulamos la media de los tf-idf de los k documentos mas relevantes
                total = {}

                for r in response:  # Calculamos la suma de los tf-idf k documentos mas relevantes
                    file_tw = toTFIDF(client,index, r.meta.id)
                    total = {t: file_tw.get(t,0) + total.get(t,0) for t in set(file_tw) | set(total)}

                for m in mostrar: #Mostramos los nhits documentos mas relevantes
                    print(f'ID= {m.meta.id} SCORE={m.meta.score}')
                    print(f'PATH= {m.path}')
                    print(f'TEXT: {m.text[:50]}')
                    print('-----------------------------------------------------------------')                    

                calc1 = {l:  alpha*v for l, v in dic.items()} ##alpha*query
                calc2 = {l: v*beta/k for l, v in total.items()} ##beta*vectorDocumentos / K
                calcTotal = {t: calc1.get(t,0) + calc2.get(t,0) for t in set(calc1) | set(calc2)} #alpha*query + beta*vectorDocumentos/K
                calcTotal = sorted(calcTotal.items(), key = lambda x: x[1], reverse=True) #Ordenamos por tf-idf
                calcTotal = calcTotal[0:R] #Cogemos las R palabras con mayor tf-idf
                nextQuery = dict((l, v) for (l,v) in calcTotal)
                query = toQuery(normalize(nextQuery)) #Convertimos en lista la nueva query obtenida

                print(query) #Mostramos la nueva query generada con Rocchio

        else:
            print('No query parameters passed')

        print (f"{response.hits.total['value']} Documents")

    except NotFoundError:
        print(f'Index {index} does not exists')
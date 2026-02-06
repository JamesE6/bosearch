import cohere
import numpy as np
from collections import defaultdict
import os
from itertools import islice

co = cohere.Client('COHERE_API_KEY')
pinecone_API_key = 'PINECONE_API_KEY'

def embed_docs(file):
    
    return co.embed(
    texts = file,
    model = 'embed-multilingual-v2.0',
    input_type = 'search_document'
    )

def embed_query(query):
    return co.embed(
    texts = [query],
    model = 'embed-multilingual-v2.0',
    input_type = 'search_query'
    )

def embed_files(list_of_dirs):
    
    all_embeddings = []
    indices = []
    
    for dir_name in list_of_dirs:
    
        for fname in os.listdir(dir_name):

            with open(os.path.join(dir_name, fname)) as f:
                testfile = f.read()
        
        testfile = testfile.split('    ')
        testfile = [line.strip() for line in testfile]
        sent_idxs = [fname[:-4] + '_' + str(i) for i in range(len(testfile))]
        new_embeddings = co.embed(testfile).embeddings

        all_embeddings.extend(new_embeddings)
        indices.extend(sent_idxs)
    
    return all_embeddings, indices
    
def calculate_similarity(a, b):
    
    if a.shape != b.shape:
        raise ValueError(f'Input vectors must have the same shape. Input was {a.shape} and {b.shape}')
        print(type(a))
        print(type(b))
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def return_results(query, embeddings, indices, n_results, scores=False):
    if not isinstance(query, str):
        raise TypeError('Query must be a string')
    
    query_embeddings = co.embed([query]).embeddings
    query_embedding = np.array(query_embeddings[0])
    
    result_dict = {}
    
    for i, doc in enumerate(embeddings):
        
        doc = np.array(doc)
        result_dict[i] = calculate_similarity(query_embedding, doc)
        
    result_dict = dict(sorted(result_dict.items(), key = lambda item: item[1], reverse = True))
    
    limited_results = dict(islice(result_dict.items(), n_results))
    
    if scores:
        
        return dict(zip([indices[i] for i in limited_results.keys()], limited_results.values()))
    
    else:
        return [indices[i] for i in limited_results.keys()]



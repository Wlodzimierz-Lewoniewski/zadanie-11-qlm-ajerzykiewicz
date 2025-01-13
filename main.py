import numpy as np
import re

def calculate_tf(t, document):
    return document.count(t)

def calculate_prob_t_given_d(tf, L_d, smoothing=False, lambda_=0.5, P_t_given_c=0):
    if smoothing:
        return lambda_ * (tf / L_d) + (1 - lambda_) * P_t_given_c
    else:
        return tf / L_d

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def calculate_query_likelihood(query, documents, corpus, lambda_=0.5):

    corpus_flat = " ".join(corpus)
    corpus_terms = tokenize(corpus_flat)
    total_terms_corpus = len(corpus_terms)

    P_t_given_c = {t: corpus_terms.count(t) / total_terms_corpus for t in set(corpus_terms)}

    query_likelihoods = {}

    for doc_idx, doc in enumerate(documents):
        doc_terms = tokenize(doc)
        L_d = len(doc_terms)
        
        P_q_given_d = 1
        for t in query:
            tf = calculate_tf(t, doc_terms)
            P_t_given_d = calculate_prob_t_given_d(tf, L_d, smoothing=True, lambda_=lambda_, P_t_given_c=P_t_given_c.get(t, 0))
            P_q_given_d *= P_t_given_d

        query_likelihoods[doc_idx] = P_q_given_d

    sorted_likelihoods = sorted(query_likelihoods.items(), key=lambda x: x[1], reverse=True)

    return [doc_idx for doc_idx, _ in sorted_likelihoods]


n = int(input("Enter the number of documents: "))
documents = [input(f"Enter document {i+1}: ") for i in range(n)]
query = tokenize(input("Enter the query: "))

corpus = documents
ranking = calculate_query_likelihood(query, documents, corpus, lambda_=0.5)
print(ranking)

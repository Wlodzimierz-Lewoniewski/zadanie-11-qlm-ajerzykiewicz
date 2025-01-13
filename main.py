import numpy as np
import re

def calculate_tf(t, document):
    return document.count(t)

def calculate_prob_t_given_d(tf, L_d):
    return tf / L_d if L_d > 0 else 0

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def calculate_query_likelihood(query, documents):

    query_likelihoods = []

    for doc_idx, doc in enumerate(documents):
        doc_terms = tokenize(doc)
        L_d = len(doc_terms)

        P_q_given_d = 1
        for t in query:
            tf = calculate_tf(t, doc_terms)
            P_t_given_d = calculate_prob_t_given_d(tf, L_d)
            P_q_given_d *= P_t_given_d

        query_likelihoods.append((doc_idx, P_q_given_d))

    sorted_likelihoods = sorted(query_likelihoods, key=lambda x: x[1], reverse=True)

    return [doc_idx for doc_idx, _ in sorted_likelihoods]

n = int(input())
documents = [input().strip() for _ in range(n)]
query = tokenize(input().strip())

ranking = calculate_query_likelihood(query, documents)

print(ranking)

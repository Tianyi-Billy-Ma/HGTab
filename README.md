WikiSQL:

Baselines:

Contriever (Pretrain):  "1": 58.52

dpr (Pretrain):         "1": 57.25

dpr + allset:           "1": 73.47

WikiTQ

Baselines:

Contriever (Pretrain):  "1": 37.26

dpr (Pretrain):         "1": 35.75

dpr + allset:           "1": 58.85


Model:

Retriever: 

question_encoder: facebook/dpr-question_encoder-single-nq-base
table_encoder: one embedding layer + allset 
objective: cl loss between question embeddings and table_embeddings (postive and negative)

*variation: use augmentation for negative samples.*

Generator(TODO):

1. sub_table retrieval (Paper ITR)
2. Sequence_to_sequence transformer model as generator to answer the question
3. Joint Training, i.e., objective from retriever + generator
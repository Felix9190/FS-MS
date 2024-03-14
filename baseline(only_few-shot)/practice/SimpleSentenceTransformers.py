import torch
from sentence_transformers import SentenceTransformer

# more correct
sentences = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted-metal-sheets', 'Bare-Soil', 'Bitumen', 'Self-Blocking-Bricks', 'Shadows']

model = SentenceTransformer('bert-base-nli-mean-tokens')

# mapping
sentences_embeddings = model.encode(sentences) # (9, 768) numpy

print(sentences_embeddings.shape[0])

sentences_embeddings_list = []


for i in range(sentences_embeddings.shape[0]): # real_class_list
    sentences_embeddings_list.append(torch.from_numpy(sentences_embeddings[i]).to(0))
print(sentences_embeddings_list[0])

x = torch.zeros(sentences_embeddings.shape[0], 768)
print(x)
x[0] = torch.from_numpy(sentences_embeddings[0])

print(x)

print(sentences_embeddings)

print(sentences_embeddings.shape) # (9, 768)

# from sklearn.metrics.pairwise import cosine_similarity
#
# print(cosine_similarity([sentences_embeddings[0]], sentences_embeddings[1:]))

sentences_embeddings = torch.from_numpy(sentences_embeddings)

from utils import utils
print(-1 * utils.euclidean_metric(sentences_embeddings, sentences_embeddings))


import gensim
from collections import Counter
from keras.preprocessing.sequence import skipgrams
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
def load_data():
	#clear data, etc
	with open('test2.txt', 'r+') as file:
		data = [[word for word in line.split() if word not in stopwords.words('english')] for line in file]
		return data

	


#list of sentences(lists)
raw_data = load_data()
flat_data = [word for text in raw_data for word in text]
#dictionary ->  word : count
unigram_counter = Counter(flat_data)


vocab_size = len(unigram_counter)
window_size = 5 	

print("vocabulary size: {}".format(vocab_size))

print("10 most common objects: {}".format(unigram_counter.most_common(10)))
vocab_index = dict()

#map object with index
it = 0
for word, _ in unigram_counter.items():
	vocab_index[word] = it
	it += 1

object_by_index = {indx:obj for obj,indx in vocab_index.items()}	

def similarity(word, matrix, n=10):
	index = vocab_index[word]
	if isinstance(matrix, csr_matrix):
		v1 = matrix.getrow(index)
	else:
		v1 = matrix[index:index+1, :]
	sims = cosine_similarity(matrix, v1).flatten()
	sindxs = np.argsort(-sims)
	sim_word_scores = [(object_by_index[sindx], sims[sindx]) for sindx in sindxs[0:n]]
	return sim_word_scores


#dictionary of skipgram with window = 4 -> words -> count
skipgram_pairs, skipgram_count = skipgrams(flat_data, vocab_size, window_size=window_size)

#to dict
skipgram_counter = Counter()
for i in range(len(skipgram_pairs)):
	if skipgram_pairs[i][0] in flat_data and skipgram_pairs[i][1] in flat_data:
		skipgram_counter[(skipgram_pairs[i][0],skipgram_pairs[i][1])] += skipgram_count[i]

print("skipgram size: {}".format(len(skipgram_counter)))

print("10 most common skipgram objects: {}".format(skipgram_counter.most_common(10)))

#skipgrams count matrix
x_indxs = []
y_indxs = []
sgc_values = []

for (object1, object2), sg_count in skipgram_counter.items():
	i_x = vocab_index[object1]
	i_y = vocab_index[object2]

	x_indxs.append(i_x)
	y_indxs.append(i_y)
	sgc_values.append(sg_count)

sgc_mx = csr_matrix((sgc_values, (x_indxs, y_indxs)))

print("Skipgram count matrix finished")

#log(skipgram(x,y)/(skipgram(x)*skipgram(y)))
# PMI 
x_indxs = []
y_indxs = []
pmi_values = []
objects_count = sgc_mx.sum()

sum_object = np.array(sgc_mx.sum(axis=0)).flatten()
sum_context = np.array(sgc_mx.sum(axis=1)).flatten()

for (object1, object2), sg_count in skipgram_counter.items():

	i_x = vocab_index[object1]
	i_y = vocab_index[object2]
	
	nwc = sg_count
	Pwc = nwc/objects_count
	nw = sum_context[i_x]
	nc = sum_context[i_y]
	Pw = nw/objects_count
	Pc = nc/objects_count

	pmi = np.log2(Pwc/(Pw*Pc))

	x_indxs.append(i_x)
	y_indxs.append(i_y)
	pmi_values.append(pmi)

pmi_mx = csr_matrix((pmi_values, (x_indxs, y_indxs)))

print("PMI matrix finished")

# SVD

embedding_size = 100

u_mx, _, v_mx = svds(pmi_mx, embedding_size)

word_embeddings = u_mx + v_mx.T

words=['people', 'get']

for word in words:
	print(similarity(word, word_embeddings, 3))



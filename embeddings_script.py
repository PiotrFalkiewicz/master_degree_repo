from gensim.models import word2vec
import pandas as pd
from search_algorithm.py import power_means
import gensim
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import log

def generate_svd(data_dict, n_dim = 300):
	data_list = [v for _,v in data_dict.items()]

	unigrams_cnt = Counter()
	bigrams_cnt = Counter()
	for text in data_list:
		for x in text:
			unigrams_cnt[x] += 1
		for x, y in map(sorted, combinations(text, 2)):
				bigrams_cnt[(x, y)] += 1

	id2uni = {}
	uni2id = {}
	it = 0

	for uni,_ in unigrams_cnt.items():
		id2uni[it] = uni
		uni2id[uni] = it
		it +=1


	sum_uni = sum(unigrams_cnt.values())
	sum_bi = sum(bigrams_cnt.values())

	pmi_samples = Counter()
	data, rows, cols = [], [], []
	for (x, y), n in bigrams_cnt.items():
		rows.append(uni2id[x])
		cols.append(uni2id[y])
		data.append(log((n / sum_bi) / (unigrams_cnt[x] / sum_uni) / (unigrams_cnt[y] / sum_uni)))
		pmi_samples[(x, y)] = data[-1]
	PMI = csc_matrix((data, (rows, cols)))

	U,_,_ = svds(PMI, k = n_dim)
	norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
	U /= np.maximum(norms, 1e-7)

	result_dict = {}

	for key in data_dict:
		result_dict[key] = power_means([np.dot(U, U[uni2id[product]]) for product in data_dict[key]])

	return result_dict


def generate_word2vec(data_dict, n_dim = 300, n_workers = 10, n_epochs = 20):
	data_list = [v for _,v in data_dict.items()]

	model = word2vec.Word2Vec(data_list, size = n_dim, min_count = 1, workers = n_workers)

	model.train(data_list, total_examples = len(data_list), epochs = n_epochs)

	result_dict = {}

	for key in data_dict:
		result_dict[key] = power_means([model[product] for product in data_dict[key]])

	return result_dict


def create_index(input_set, d = 8):
	flatten_input = list(set([ j for i in input_set.values() for j in i]))
	index_dict = {}
	l = int(np.floor(math.log(len(flatten_input), d))) + 1

	# print("Index depth is gonna be {} (including root level) with d = {} and length of vocabulary = {}".format(l,d,len(flatten_input)))


	# print("Size of index base = {}".format(d**l))

	bitmap_index_size = 0

	for i in range(1,l+1):
		bitmap_index_size += d**i

	# print("Length of bitmap index = {}".format(bitmap_index_size))


	for i in range(len(flatten_input)):
		index_dict[flatten_input[i]] = [1 if j == i else 0 for j in range(d**l)]

	# print(index_dict)

	bitmap_index = [[0 for i in range(d)] for i in range(int(bitmap_index_size/d))]

	# print(bitmap_index)

	result_dict = {}


	for key in input_set.keys():
		index = [0 for i in range(d**l)]
		for i in input_set[key]:
			index = [x+y for x,y in zip(index,index_dict[i])]
		
		last_level_index = index

		final_index = []
		final_index.append(index)

		while len(reduce_level(last_level_index,d)) >= d:
			temp_index = reduce_level(last_level_index,d)
			final_index = [temp_index] + final_index
			last_level_index = temp_index

		final_index = reduce_index(final_index, d)

		result_dict[key] = final_index


	#return dict with same keys but with index
	return result_dict

def reduce_level(bits, d):
	l = int(len(bits)/d)
	result = []

	for i in range(l):
		if sum(bits[i*d:(i+1)*d]) > 0:
			result.append(1)
		else:
			result.append(0)

	return(result)


def reduce_index(index, d):
	result = []

	for sublist in index:
		if len(sublist) > d:
			for i in range(int(len(sublist)/d)):
				result.append(sublist[i*d:(i+1)*d])
		else:
			result.append(sublist)

	final_result = []

	for r_list in result:
		if sum(r_list) > 0:
			final_result.append(r_list)


	return(final_result)
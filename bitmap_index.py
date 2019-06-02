import pandas as pd
import numpy as np
import math

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



def create_index(input_set, d = 8):
	flatten_input = list(set([ j for i in input_set.values() for j in i]))
	index_dict = {}
	l = int(np.floor(math.log(len(flatten_input), d))) + 1

	print("Index depth is gonna be {} (including root level) with d = {} and length of vocabulary = {}".format(l,d,len(flatten_input)))


	print("Size of index base = {}".format(d**l))

	bitmap_index_size = 0

	for i in range(1,l+1):
		bitmap_index_size += d**i

	print("Length of bitmap index = {}".format(bitmap_index_size))


	for i in range(len(flatten_input)):
		index_dict[flatten_input[i]] = [1 if j == i else 0 for j in range(d**l)]

	# print(index_dict)

	bitmap_index = [[0 for i in range(d)] for i in range(int(bitmap_index_size/d))]

	# print(bitmap_index)

	index_set = {}


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

		index_set[key] = final_index


	#return dict with same keys but with index
	return index_set


input = [['A','B'],['A','B','C'],['A','B','C','D','E'],['F','G','H','Z','L','M'],['N','O','P'],['AA','BB','CC','DD','EE','FF','GSDF','KYT','ADS'],['Fds']]
# input = [['A','B'],['A','B','C']]
input_as_dict = {}

for i in range(len(input)):
	input_as_dict[i] = input[i]



indexed_input = create_index(input_as_dict)

print(indexed_input)


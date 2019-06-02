import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
import math

def cosine_distance(a,b):
	return 1.0 - np.dot(a, b)/(norm(a)*norm(b)) 



def a_nn(object_emb, data_dict, cutoff=0.3, k = 3):
	result = []
	for key in data_dict.keys():

		distance = cosine_distance(object_emb, data_dict[key])

		if distance < cutoff:
			result.append(key)

		if len(result) == k:
			break

	return result


def calc_cutoff(objects_dict, emb_dict, precision = 0.0001, value = 0.5, gamma = 0.01, iters = 0.3, n_iters = 0):
	keys = [key for key in objects_dict.keys()]
	max_iter = int(iters*len(keys))

	df = lambda x: x - (x**2)/2

	if n_iters > 0:
		max_iter = max(len(keys),n_iter)

	for i in range(max_iter):
		get_key = random.choice(keys)

		current = value

		for key in keys:

			if get_key != key:

				if set(objects_dict[get_key]) <= set(objects_dict[key]):
					distance = cosine_distance(emb_dict[get_key],emb_dict[key])

					value = current - gamma * df(distance)

					break

		if np.abs(current - value) < precision:

			break

	return value


def power_means(list_of_vectors):
    temp1,temp2,temp3 = [],[],[]
    for it in range(len(list_of_vectors[0])):
        tempC1 = 0
        tempC2 = []
        tempC3 = 0
        for vector in list_of_vectors:
            tempC1 += vector[it]
            tempC2.append(vector[it])
            tempC3 += 1/vector[it]
        temp1.append(tempC1/len(vector))
        temp2.append((np.sign(tempC2)*(np.abs(np.prod(tempC2)))**(1/len(tempC2)))[0])
        temp3.append(len(vector)/tempC3)
    
    return temp1 + temp2 + temp3












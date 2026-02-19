import time
from sklearn.neighbors import KNeighborsClassifier

def KNN(train_data, testing_data, train_labels, **params_dict):

	def drop_duplicate(data, labels):
		data_eli = []
		label_eli = []
		for idx, x in enumerate(data):
			if x not in data_eli:
				data_eli.append(x)
				label_eli.append(labels[idx])
		return data_eli, label_eli
	start_time = time.time()
	train_data, train_labels_new = drop_duplicate(train_data, train_labels)

	#training
	clf = KNeighborsClassifier(**params_dict)
	clf.fit(train_data, train_labels_new)

	end_time = time.time()
	train_time = end_time-start_time

	start_time = time.time()
	pred = []
	pre_dict = {}
	for idx, x in enumerate(testing_data):
		if str(x) not in pre_dict.keys():
			temp_prediction = list(clf.predict([x]))[0]
			pre_dict[str(x)] = temp_prediction
			pred.append(temp_prediction)
		else:
			pred.append(pre_dict[str(x)])
	end_time = time.time()
	infer_time = end_time-start_time
	return pred, train_time, infer_time

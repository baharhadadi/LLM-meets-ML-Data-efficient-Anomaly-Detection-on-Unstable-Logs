import time
from sklearn.neural_network import MLPClassifier

def MLP(train_data, testing_data, train_labels, **params_dict):
	start_time = time.time()

	#training
	clf = MLPClassifier(**params_dict)
	clf.fit(train_data, train_labels)

	end_time = time.time()
	train_time = end_time-start_time

	start_time = time.time()
	pred = []
	for t in testing_data:
		pred +=list(clf.predict([t]))
	end_time = time.time()
    
	infer_time = (end_time-start_time)/len(testing_data)
	return pred, train_time, infer_time
import time
from sklearn import tree

def decision_tree(train_data, testing_data, train_labels, **params_dict):

	start_time = time.time()

	#training
	clf = tree.DecisionTreeClassifier(**params_dict)
	clf = clf.fit(train_data, train_labels)

	end_time = time.time()
	train_time = end_time-start_time

	# Corrected the indentation of the following lines
	# DOT data
	#dot_data = tree.export_graphviz(clf, out_file=None,
    #                              feature_names=feature_names,
    #                              class_names=["normal", "anomalous"], # Assuming 'iris' is defined
    #                              filled=True)

	# Draw graph
	#graph = graphviz.Source(dot_data, format="png")
	#graph.render("decision_tree_graphivz")
	start_time = time.time()
	pred = []
	for t in testing_data :
		pred+=list(clf.predict([t]))
	end_time = time.time()
	infer_time = (end_time-start_time)/len(testing_data)

	return pred, train_time, infer_time

import pickle

def data_loader(train_path, test_path, dedup = True):

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    with open(test_data, "rb") as f:
        test_data = pickle.load(f)

    if dedup == True:

        unique_seqs = set()
        for t in train_data:
            unique_seqs.add(tuple(t["EventId"]))

        dedup_test = []
        for t in test_data:
            if tuple(t["EventId"]) not in unique_seqs:
                dedup_test.append(t)

        test_data = dedup_test

    return train_data, test_data

        
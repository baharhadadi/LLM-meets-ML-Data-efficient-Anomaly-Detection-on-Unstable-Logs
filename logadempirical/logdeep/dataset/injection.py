import os
import re
import pandas as pd
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np
import pickle

def inject_window(data_dir, inject_path, injection_rate=10, inject_type= "all"):

    # read test_window and with injection
    with open(data_dir, "rb") as input_file:
      test_window = pickle.load(input_file)

    positive_test = [i for i, x in enumerate(test_window) if x["Label"]]
    negative_test = [i for i, x in enumerate(test_window) if x["Label"]==0]

    random.seed(0)
    test_window_pos = random.sample(positive_test,1000)

    random.seed(0)
    test_window_neg = random.sample(negative_test,50000)

    test_window_new = [test_window[i] for i in test_window_pos]+[test_window[i] for i in test_window_neg]

    random.seed(0)
    random.shuffle(test_window_new)
    test_window = test_window_new

    with open(inject_path, "rb") as input_file:
      unstable_log = pickle.load(input_file)
    
    results = []

    counter = 0

    if inject_type == "all":

      for i, element in enumerate(test_window):
        if unstable_log[i]["type"] =="stable" or injection_rate == 0:
          results.append(element)

        elif injection_rate==5 and counter%6==0:
          results.append(unstable_log[i])
                  
        elif injection_rate==10 and counter%3==0:
          results.append(unstable_log[i])

        elif injection_rate==15 and counter%2==0:
          results.append(unstable_log[i])
                  
        elif injection_rate==20 and counter%3<2:
          results.append(unstable_log[i])          

        elif injection_rate==25 and counter%6<5:
          results.append(unstable_log[i])
        
        elif injection_rate==30:
          results.append(unstable_log[i])
          
        else:
          results.append(element)
        counter+=1

    else:
      
      for i, element in enumerate(test_window):
        if unstable_log[i]["type"] != inject_type:
          results.append(element)

        else:
          results.append(element)
          counter +=1

    with open(data_dir, "wb") as input_file:
      pickle.dump(results, input_file)

    del results
    del test_window
    del unstable_log

    return

def inject_type_window(test_window, unstable_type = "shuffle"):
    rate = 30/100

    idx = random.sample_without_replacement(np.arange(len(test_window)), int(len(test_window)*rate), random_state=0)

    results = []

    # read unstable log 


    for i, element in enumerate(test_window):
        if i not in idx:
          results.append(element)

        else:
          if counter < len(unstable_log):
            if unstable_log[counter]["type"] == unstable_type:
              results.append(unstable_log[counter])
              counter +=1
          else:
            print("Error the number of unstable logs does not match the number of requaired injection rate")

    return results

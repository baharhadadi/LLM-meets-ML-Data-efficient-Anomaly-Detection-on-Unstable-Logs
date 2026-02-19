import torch
import pickle
import pandas as pd
import random
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
from argparse import ArgumentParser
import time
from sklearn.metrics import f1_score, recall_score, precision_score
import os
import random
import numpy as np

prompt_description_mistral = """<s>[INST]Below is an instruction that describes a task, paired with an input. Generate an output that appropriately completes the request.

### Instruction
Given a log sequence delimited by brackets, generate the word "normal" if the sequence is normal; otherwise, "anomalous". Anomalous sequences are usually associated with unlikely sequences or sequences indicating errors, problems, or faults. No explanation is required.

### Relevant Information
Please label the input based on the following information on system calls:
{}

### Input
log sequence: {}

### Output
label: [/INST]{}</s>"""

prompt_no_description_mistral = """<s>[INST]Below is an instruction that describes a task, paired with an input. Generate an output that appropriately completes the request.

### Instruction
Given a log sequence delimited by brackets, generate the word "normal" if the sequence is normal; otherwise, "anomalous". Anomalous sequences are usually associated with unlikely sequences or sequences indicating errors, problems, or faults. No explanation is required.

### Input
log sequence: {}

### Output
label: [/INST]{}</s>"""

prompt_description_llama = """Below is an instruction that describes a task, paired with an input. Generate an output that appropriately completes the request.

### Instruction
Given a log sequence delimited by brackets, generate the word "normal" if the sequence is normal; otherwise, "anomalous". Anomalous sequences are usually associated with unlikely sequences or sequences indicating errors, problems, or faults. No explanation is required.

### Relevant Information
Please label the input based on the following information on system calls:
{}
\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n
### Input
log sequence: {}

### Output
label: {}"""

prompt_no_description_llama = """Below is an instruction that describes a task, paired with an input. Generate an output that appropriately completes the request.

### Instruction
Given a log sequence delimited by brackets, generate the word "normal" if the sequence is normal; otherwise, "anomalous". Anomalous sequences are usually associated with unlikely sequences or sequences indicating errors, problems, or faults. No explanation is required.

### Input
log sequence: {}

### Output
label: {}"""

class LogSequenceModel:
    def __init__(self, mode, model_name, train_set_path, test_set_path, templates_file, description_file, results_file, max_length, prediction, max_steps=600, output_dir = "output", save_output=None, threshold = 0.8):
        """
        Initialize the LogSequenceModel with model, dataset, and training parameters.
        """
        self.model_name = model_name
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path
        self.templates_file = templates_file
        self.description_file = description_file
        self.results_file = results_file
        self.max_steps = max_steps
        self.max_seq_length = max_length  # Maximum sequence length for input data
        self.dtype = None  # Automatically detect dtype
        self.load_in_4bit = True  # Whether to load the model using 4-bit quantization for reduced memory usage
        self.model, self.tokenizer = self.load_model()
        self.threshold = threshold
        self.prediction = prediction

        if mode=="train":
            self.model = self.config_perf()

        self.output_dir = output_dir if output_dir != "output" else self.model_name+"-output"
        self.save_output = self.output_dir if save_output == None else save_output

        templates = pd.read_csv(self.templates_file)
        self.templates = dict(zip(templates['EventId'], templates['EventTemplate']))

    def load_model(self):
        """
        Load the pre-trained model and its tokenizer from Unsloth.
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

        return model, tokenizer

    def save_model(self):
        """
        Save the trained model to the specified output directory.
        """
        self.model.save_pretrained(self.save_output) # Local saving
        self.tokenizer.save_pretrained(self.save_output) # Local saving

    def config_perf(self):
        # Configure the model using PEFT (Parameter-Efficient Fine-Tuning)
        model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Disable dropout for LoRA (Low-Rank Adaptation)
            bias="none",  # Optimized configuration
            use_gradient_checkpointing="unsloth",  # Efficient gradient checkpointing #random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return model

    def load_dataset(self):
        """
        Load the dataset from the provided training set path using pickle.
        """
        with open(self.train_set_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def pkl_to_dict(self, file_path):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
        return d

    def xlsx_to_dict(self, file_path):
        """
        Convert an Excel file (xlsx) to a dictionary mapping syscall names to descriptions.
        """
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            # Create a dictionary from the first and third columns
            first_column = df.iloc[:, 1].tolist()
            third_column = df.iloc[:, 2].tolist()
            data_dict = dict(zip(first_column, third_column))
            return data_dict
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def formatting_prompts_func(self, ft_dataset, my_dict, llama_prompt):
        """
        Format the input dataset into prompts using system call descriptions and a pre-defined template.
        """
        texts = []
        for d in ft_dataset:
            input_seq = [self.templates[i] for i in d["EventId"]]
            output = "normal" if d["Label"] == 0 else "anomalous"

            if my_dict != None:
              syscall_descriptions = ""

              # Add syscall descriptions from the dictionary
              for syscall in input_seq:
                  syscall = syscall.replace("__NR_", "")
                  if syscall in my_dict.keys() and syscall not in syscall_descriptions:
                      syscall_descriptions += f"{syscall}: {my_dict[syscall]}\n"

              # Format the final prompt using the template
              prompt = llama_prompt.format(syscall_descriptions, input_seq, output)
              texts.append({"text": prompt})

            else:
              prompt = llama_prompt.format(input_seq, output)
              texts.append({"text": prompt})

        return texts

    def train_model(self):
        """
        Train the model using the dataset and parameters.
        """
        # Load and prepare the dataset
        data = self.load_dataset()
        #selected_data = self.select_samples_without_replacement(data)
        selected_data = data

        # Convert system call descriptions to a dictionary
        if self.description_file != None:
            my_dict = self.pkl_to_dict(self.description_file)

            # Format dataset into prompts
            if "Mistral" in self.model_name:
                dataset = self.formatting_prompts_func(selected_data, my_dict, prompt_description_mistral)
                dataset = Dataset.from_list(dataset)
            elif "Llama" in self.model_name:
                dataset = self.formatting_prompts_func(selected_data, my_dict, prompt_description_llama)
                dataset = Dataset.from_list(dataset)
            else:
                print("Error: a dataset with mistral name is asked but it was part of accepted models")

        else:
            
            if "Mistral" in self.model_name:
                dataset = self.formatting_prompts_func(selected_data, None, prompt_no_description_mistral)
                dataset = Dataset.from_list(dataset)
            elif "Llama" in self.model_name:
                dataset = self.formatting_prompts_func(selected_data, None, prompt_no_description_llama)
                dataset = Dataset.from_list(dataset)
            else:
                print("Error: a dataset with mistral name is asked but it was part of accepted models")

        # Define the training arguments and initialize the trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Set to False to prevent packing multiple sequences into the same input
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=self.max_steps,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",  # Optimizer used for training
                weight_decay=0.01,
                lr_scheduler_type="linear",
                output_dir=self.output_dir,  # Directory to save outputs
            ),
        )

        # Print GPU information
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # Start training and get statistics
        trainer_stats = trainer.train()

        # Print memory usage and training time statistics
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # save model and tokenizer
        self.save_model()

    def inference(self):
        """
        Perform inference on the test dataset and compute metrics such as F1-score, recall, and precision.
        """
        # Prepare to store results
        results = {}
        predict = []
        actual = []
        elapsed_times = []

        # Switch the model to inference mode
        FastLanguageModel.for_inference(self.model)

        # Load the test data
        with open(self.test_set_path, 'rb') as f:
            self.test_data = pickle.load(f)

        # if the descriptions are given
        if self.description_file != None:
            my_dict = self.pkl_to_dict(self.description_file)
            if "mistral" in self.model_name:
                llama_prompt = prompt_description_mistral
            elif "llama" in self.model_name:
                llama_prompt = prompt_description_llama
            else:
                print("unrecognized model")
        else:
            if "mistral" in self.model_name:
                llama_prompt = prompt_no_description_mistral
            elif "llama" in self.model_name:
                llama_prompt = prompt_no_description_llama
            else:
                print("unrecognized model")

        high_conf_wrong_predict = []
        high_conf_correct_predict = []
        low_conf_wrong_predict = []
        low_conf_correct_predict = []

        # Iterate through the test data and perform inference
        for j, log in enumerate(self.test_data):
            print("test unit: ", j, "Label: ", "anomalous" if log["Label"] else "normal" )
            actual.append(log["Label"])

            # Check if results for this log are cached
            if tuple(log["EventId"]) in results.keys():
                start_time = time.time()
                predict.append(results[tuple(log["EventId"])][0])
                elapsed_time = time.time() - start_time
                elapsed_times.append(elapsed_time)
                #self.prediction.append({"seq":log["seq"], "actual":log["Label"], "predict":results[tuple(log["EventId"])][0]})

                if results[tuple(log["EventId"])][2] <self.threshold:
                    if results[tuple(log["EventId"])][0] != log["Label"]:
                        low_conf_wrong_predict.append(j)
                    else:
                        low_conf_correct_predict.append(j)

                else:
                    if results[tuple(log["EventId"])][0] != log["Label"]:
                        high_conf_wrong_predict.append(j)
                    else:
                        high_conf_correct_predict.append(j)
            else:
                input_seq = [self.templates[i] for i in log["EventId"]]

                if self.description_file != None:
                  syscall_descriptions = ""
                  for syscall in input_seq:
                      syscall = syscall.replace("__NR_","")
                      if syscall in my_dict.keys() and syscall not in syscall_descriptions:
                        syscall_descriptions += f"{syscall}: {my_dict[syscall]}\n"

                  prompt = llama_prompt.format(syscall_descriptions, input_seq, "")

                else:
                  prompt = llama_prompt.format(input_seq, "")

                a = log["Label"]

                # Format the input using the template
                inputs = self.tokenizer(
                    [
                        prompt
                    ], return_tensors="pt"
                ).to("cuda")

                # Perform inference
                start_time = time.time()
                text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                print("output limited new tokens")
                outputs = self.model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, streamer = text_streamer, max_new_tokens = 3, temperature=0.1, output_logits=True, output_scores=True, return_dict_in_generate=True)
                elapsed_time = time.time() - start_time

                t = self.tokenizer.batch_decode(outputs["sequences"])
                t[0] = t[0].replace("\n"," ").replace("<|end_of_text|>"," ").replace("<|eot_id|>"," ").replace("</s>"," ")
                while t[0].strip()=='' or len(t[0].split("label: ", 1)[-1].strip().split())==0:
                  print("empty output")
                  outputs = self.model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, streamer = text_streamer, max_new_tokens = 5, temperature=1, output_logits=True, output_scores=True, return_dict_in_generate=True)
                  t = self.tokenizer.batch_decode(outputs["sequences"])
                  t[0] = t[0].replace("\n"," ").replace("<|end_of_text|>"," ").replace("<|eot_id|>"," ")

                normal_confidence, anomalous_confidence = self.get_confidence(outputs.logits)

                prediction = "anomalous" if normal_confidence < anomalous_confidence else "normal"
                prediction_value = 0 if normal_confidence > anomalous_confidence else 1

                if self.decode_output(self.tokenizer.batch_decode(outputs["sequences"]), inputs, text_streamer) != prediction_value:
                  print("*** wrong confidence score calculation!!")

                  if self.decode_output(self.tokenizer.batch_decode(outputs["sequences"]), inputs, text_streamer)==0:
                    pass
                    normal_confidence, anomalous_confidence = anomalous_confidence-0.3, normal_confidence+0.3

                  logit = outputs.logits[0]
                  probs = torch.nn.functional.softmax(logit, dim=-1)

                  # keep only the top 20
                  probs, ids = torch.topk(probs, 20)

                  # convert ids to tokens
                  texts = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())

                  # print
                  dictionary = {}
                  for prob, text in zip(probs[0], texts):
                      text = self.tokenizer.convert_tokens_to_string([text])
                      dictionary[text] = prob.item()
                      print(f"{prob:.4f}: \"{text}\"")

                if max(normal_confidence, anomalous_confidence) < self.threshold or normal_confidence < anomalous_confidence <0.9:

                  normal_confidence_second, anomalous_confidence_second = normal_confidence, anomalous_confidence

                  if normal_confidence_second >= anomalous_confidence_second:
                    p = 0
                  else:
                    p = 1

                  if a!= p and max(normal_confidence_second, anomalous_confidence_second)< self.threshold:
                    print("low confidence wrong prediction")
                    low_conf_wrong_predict.append(j)
                  elif a== p and max(normal_confidence_second, anomalous_confidence_second)< self.threshold:
                    print("low confidence correct prediction")
                    low_conf_correct_predict.append(j)
                  elif a!= p and max(normal_confidence_second, anomalous_confidence_second)>= self.threshold:
                    print("high confidence wrong prediction")
                    high_conf_wrong_predict.append(j)
                  else:
                    print("high confidence correct prediction")
                    high_conf_correct_predict.append(j)

                  #normal_confidence, anomalous_confidence = normal_confidence_second, anomalous_confidence_second


                else:
                  normal_confidence_second, anomalous_confidence_second = normal_confidence, anomalous_confidence
                  if normal_confidence > anomalous_confidence:
                    p = 0
                  else:
                    p = 1

                  if a!= p:
                    print("high confidence wrong prediction")
                    high_conf_wrong_predict.append(j)
                  else:
                    print("high confidence correct prediction")
                    high_conf_correct_predict.append(j)

                # Decode the model's output
                # t = self.tokenizer.batch_decode(outputs)
                # p = self.decode_output(t, inputs, text_streamer)

                p = self.decode_output(self.tokenizer.batch_decode(outputs["sequences"]), inputs, text_streamer)
                predict.append(p)
                elapsed_times.append(elapsed_time)
                #self.prediction.append({"seq":log["seq"], "actual":log["Label"], "predict":p})

                results[tuple(log["EventId"])] = [p, elapsed_time, normal_confidence, anomalous_confidence, normal_confidence_second, anomalous_confidence_second]

        # Compute evaluation metrics
        f1 = f1_score(actual, predict)
        recall = recall_score(actual, predict)
        precision = precision_score(actual, predict)

        print(f"F1-score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"High confidence wrong prediction: {self.statis(high_conf_wrong_predict)}")
        print(f"High confidence correct prediction: {self.statis(high_conf_correct_predict)}")
        print(f"Low confidence wrong prediction: {self.statis(low_conf_wrong_predict)}")
        print(f"Low confidence correct prediction: {self.statis(low_conf_correct_predict)}")

        # Calculate average time taken per test log
        total_time = sum(elapsed_times)
        average_time = total_time / len(self.test_data)
        print(f"Average elapsed time: {average_time}")

        # check if the file doesn't exit, make a new one with header
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                f.write("Model, Dataset, F1-score, Recall, Precision, Avg Elapsed, high conf correct pred, low conf correct pred, high conf wrong pred, low conf high pred\n")

        # save results in prediction file
        with open(self.prediction, 'wb') as f:
            pickle.dump(results, f)

        # save results in results csv file as a new line
        test_set_path = self.test_set_path.split("/")[-1]
        model_name = self.model_name.split("/")[-1]
        with open(self.results_file, 'a') as f:
            f.write(f"{model_name}, {test_set_path}, {round(f1, 4)},{round(recall, 4)},{round(precision, 4)}, {average_time}, {self.statis(high_conf_correct_predict)}, {self.statis(low_conf_correct_predict)}, {self.statis(high_conf_wrong_predict)}, {self.statis(low_conf_wrong_predict)}\n")

    def statis(self, conf):
        positive = 0
        negative = 0
        for i in conf:
          if self.test_data[i]["Label"]:
            positive += 1
          else:
            negative += 1
        return (positive, negative)
    def get_confidence(self, logits):
        logit = logits[0]
        probs = torch.nn.functional.softmax(logit, dim=-1)

        # keep only the top 20
        probs, ids = torch.topk(probs, 20)

        # convert ids to tokens
        texts = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())

        dictionary = {}
        for prob, text in zip(probs[0], texts):
            text = self.tokenizer.convert_tokens_to_string([text])
            dictionary[text] = prob.item()
            #print(f"{prob:.4f}: \"{text}\"")

        p_normal = sum([dictionary[k] for k in dictionary.keys() if k in ["normal", "0", " 0", " normal", "Normal", ".normal", "-normal", "_normal"]])
        p_anomalous = sum([dictionary[k] for k in dictionary.keys() if k in [" anom", "1", " 1", "anom", " anomal", "anomal", "an", "An"]])
        if p_normal + p_anomalous != 0:
            confidence_normal = p_normal / (p_normal + p_anomalous)
            confidence_anomalous = p_anomalous / (p_normal + p_anomalous)
        else:
            confidence_normal, confidence_anomalous = 0, 0
        return confidence_normal, confidence_anomalous

    def decode_output(self, t, inputs, text_streamer):
        """
        Helper function to decode the model output and map it to 'normal' or 'anomalous' labels.
        """
        t[0] = t[0].replace("\n"," ").replace("<|end_of_text|>"," ").replace("<|eot_id|>"," ").replace("</s>"," ")
        while t[0].strip()=='' or len(t[0].split("label: ", 1)[-1].strip().split())==0:
          print("empty output")
          outputs = self.model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, streamer = text_streamer, max_new_tokens = 5, temperature=1)
          t = self.tokenizer.batch_decode(outputs)
          t[0] = t[0].replace("\n"," ").replace("<|end_of_text|>"," ").replace("<|eot_id|>"," ")

        p = -1
        if t[0].split("label: ", 1)[-1].strip().split()[0] == 'normal' or '0' == t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 0
        elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'anomalous' or '1' == t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 1
        elif 'anomalous' in t[0].split("label: ", 1)[-1].strip().split()[0] and 'normal' not in t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 1
        elif 'normal' in t[0].split("label: ", 1)[-1].strip().split()[0] and 'anomalous' not in t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 0
        elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'normal' or '0' == t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 0
        elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'anomalous' or '1' == t[0].split("label: ", 1)[-1].strip().split()[0]:
          p = 1
        elif 'anomalous' in t[0].split("label: ", 1)[-1].strip().split() and 'normal' not in t[0].split("label: ", 1)[-1].strip().split():
          p = 1
        elif 'normal' in t[0].split("label: ", 1)[-1].strip().split() and 'anomalous' not in t[0].split("label: ", 1)[-1].strip().split():
          p = 0
        else:
          k = 0
          temp = 0.1
          while p != 0 or p !=1:
            outputs = self.model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask, streamer = text_streamer, max_new_tokens = 5, temperature=1)
            t = self.tokenizer.batch_decode(outputs)
            t[0] = t[0].replace("\n"," ").replace("<|end_of_text|>"," ").replace("<|eot_id|>"," ").replace("</s>"," ")
            if t[0].strip() in ["", " "] or len(t[0].split("label: ", 1)[-1].strip().split())==0:
              print("empty output")
              continue
            if t[0].split("label: ", 1)[-1].strip().split()[0] == 'normal' or '0' == t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 0
              break
            elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'anomalous' or '1' == t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 1
              break
            elif 'anomalous' in t[0].split("label: ", 1)[-1].strip().split()[0] and 'normal' not in t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 1
              break
            elif 'normal' in t[0].split("label: ", 1)[-1].strip().split()[0] and 'anomalous' not in t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 0
              break
            elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'normal' or '0' == t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 0
              break
            elif t[0].split("label: ", 1)[-1].strip().split()[0] == 'anomalous' or '1' == t[0].split("label: ", 1)[-1].strip().split()[0]:
              p = 1
              break
            elif 'anomalous' in t[0].split("label: ", 1)[-1].strip().split() and 'normal' not in t[0].split("label: ", 1)[-1].strip().split():
              p = 1
              break
            elif 'normal' in t[0].split("label: ", 1)[-1].strip().split() and 'anomalous' not in t[0].split("label: ", 1)[-1].strip().split():
              p = 0
              break
            k+=1
            if k >= 2:
              print("long", k, t[0].split("label: ", 1)[-1].strip().split("<|end-of-text|>")[0])
              p = 0
              temp = 0.1
              break
        return p

    def load_templates(self):
        """
        Load the prompt template from the given file.
        """
        with open(self.templates_file, 'r') as file:
            template = file.read()
        return template

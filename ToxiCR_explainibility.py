# %%
#source = https://mccormickml.com/2019/07/22/BERT-fine-tuning/ 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# %%
#from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pandas as pd 
import numpy as np
import re

# %%
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# %%
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# %%
from imblearn import under_sampling


# %%
data=pd.read_excel('code-review-dataset-full.xlsx')

# %%
#data.is_toxic.value_counts()

# %%
#data=data.sample(n=1500)

# %%
data['message'] = data['message'].astype(str)

# %%
texts=data["message"]

# %%
labels=data["is_toxic"]

# %%


# %%
data.is_toxic.value_counts()

# %%
target_names = list(set(labels))
label2idx = {label: idx for idx, label in enumerate(target_names)}
#print(label2idx)

# %%

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# %%
max_len = 0

# For every sentence...
for sent in texts:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

# %%
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in texts:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
#labels = torch.tensor(labels.values)
labels = torch.tensor(labels.values)

# Print sentence 0, now as a list of IDs.
#print('Original: ', texts[0])
#print('Token IDs:', input_ids[0])

# %%
from torch.utils.data import TensorDataset, random_split
# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

print(len(dataset))



# %%
print(attention_masks[0])

# %%
train_size = int(0.9 * len(dataset))
#test_size=  int(0.2 * len(dataset))
val_size = len(dataset) - train_size

rest_data,test_data= random_split(dataset, [train_size,val_size])
print(len(test_data))
print(len(rest_data))

# %%
train_size = int(0.9 * len(rest_data))
#test_size=  int(0.2 * len(dataset))
val_size = len(rest_data) - train_size

train_data,val_data= random_split(rest_data, [train_size,val_size])
print(len(train_data))
print(len(val_data))

# %%
'''
train_size = int(0.9 * len(rest_data))
#test_size=  int(0.2 * len(dataset))
val_size = len(rest_data) - train_size

train_data,val_data=random_split(rest_data, [train_size,val_size])

print(len(train_data))
print(len(val_data))
'''

# %%
# Print the original sentence.
print(' Original: ', texts[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(texts[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[0])))

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_data,  # The training samples.
            sampler = RandomSampler(train_data), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_data, # The validation samples.
            sampler = SequentialSampler(val_data), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# %%
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification

# %%

#from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions =True, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    return_dict=False,
)

# Tell pytorch to run this model on the GPU.
model.cuda()


# %%
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# %%
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 10

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# %%
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# %%
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# %%


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# %%
loss_history = []
no_improvement = 0

# %%
PATIENCE = 2

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
MODEL_FILE_NAME="bert_model_toxic.pt"
OUTPUT_DIR = './model_save/'

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` conts:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits, attentions = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits,_) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
    
    #validation
    
    print("Loss history:", loss_history)
    print("Dev loss:", avg_val_loss)
    
    if len(loss_history) == 0 or avg_val_loss < min(loss_history):
        no_improvement = 0
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        no_improvement += 1
    
    if no_improvement >= PATIENCE: 
        print("No improvement on development set. Finish training.")
        break
        
    
    loss_history.append(avg_val_loss)
    
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# %%
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

tokenizer.save_pretrained(OUTPUT_DIR)

# %%
model.load_state_dict(torch.load('./model_save/bert_model_toxic.pt'),) # path of your weights
#model=torch.load('./model_save/bert_model_toxic.pt')


# %%

#tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR)

# Copy the model to the GPU.
model.to(device)
model.eval()


# %%
# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
#prediction_data = val_data
prediction_sampler = SequentialSampler(test_data)
prediction_dataloader = DataLoader(test_data, sampler=prediction_sampler, batch_size=batch_size)

# %%
print(len(prediction_dataloader))

# %%
print(label_ids)

# %%
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(b_input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  pp=F.softmax(logits, dim=-1).cpu().numpy()
  #print(predictions[:,1])
  
  threshold=0.5
  preds=np.where(pp[:,1]>=threshold,1,0)
  #print(preds)
  #print(F.softmax(logits, dim=-1).cpu().numpy())
  #print( torch.sigmoid(logits))
  # Move logits and labels to CPU
  #logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions+=list(preds)
  true_labels+=list(label_ids)

print('    DONE.')


# %%
predictions=np.array(predictions)
true_labels=np.array(true_labels)

# %%
len(predictions)

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score


# %%
acc=accuracy_score(true_labels,predictions)

# %%
print(acc)

# %%
f_score=f1_score(true_labels,predictions,pos_label=1)

# %%
print(f_score)

# %%
prec=precision_score(true_labels,predictions,pos_label=1)
recall=recall_score(true_labels,predictions,pos_label=1)

# %%
print(prec,recall)

# %%
# load tokenizer
tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)

# %%
##visualization

# %%


# %%
model.to(device)
inputs = tokenizer.encode("I fuck and suck the code", return_tensors='pt')
#inputs
outputs = model(inputs.to(device))
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

# %%
logits=outputs[0]

# %%
print(logits)

# %%
score=logits.softmax(dim=1)
print(score)

# %%
s=score[0][1].detach().cpu().numpy()

# %%
s

# %%

from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

# %%
from captum.attr import (
    visualization
)

# %%
classifications = ["nontoxic", "toxic"]

# %%
sentences=["go fuck yourself",
           "this is crap",
           "thank you for the information",
           "yeah that sucked, fixed, Done.",
           "Crap, this is an artifact of a previous revision. It's simply the last time a change was made to Tuskar's cloud.",
           "Ah damn I misread the bug -_-",
           "wtf...",
           "I appreciate your help.",
           "fuuuuck",
           "what the f*ck",
           "absolute shit",
           "Get the hell outta here",
            "shi*tty code",
           "you are an absolute b!tch",
           "Nothing particular to worry about"]

# %%
# encode a sentence
#text_batch = ["I fuck and suck the code","Your code is a stupid"]
#text_batch = ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great."]
#input_ids=[[]]
#attention_mask=[[]]
'''
explanations = Generator(model)
for i in range(len(sentences)):
    encoding = tokenizer(sentences[i], padding=False, return_tensors='pt')
    print(encoding['input_ids'].to("cuda"))
    print(encoding['attention_mask'].to("cuda"))
    input_ids =encoding['input_ids'].to("cuda")
    attention_mask = encoding['attention_mask'].to("cuda")
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    #expl= explanations.generate_LRP_last_layer(input_ids, attention_mask)
    expl = (expl - expl.min()) / (expl.max() - expl.min())

    # get the model classification
    output = torch.nn.functional.softmax(model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
    classification = output.argmax(dim=-1).item()
    class_name = classifications[classification]
    #print(sentences[i], "---is predicted: --", class_name)
    if class_name == "toxic":
      expl *= (-1)
    if class_name == "toxic":
        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
        print([(tokens[i], expl[i].item()) for i in range(len(tokens))])
    
'''  

# %%
'''
vis_data_records = [visualization.VisualizationDataRecord(
                                expl[0],
                                output[0][classification],
                                classification,
                                1,
                                1,
                                1,       
                                tokens,
                                1)]
visualization.visualize_text(vis_data_records)
'''

# %%


explanations = Generator(model)
for i in range(len(sentences)):
    encoding = tokenizer(sentences[i], return_tensors='pt')
    #print(encoding['input_ids'].to("cuda"))
    #print(encoding['attention_mask'].to("cuda"))
    input_ids =encoding['input_ids'].to("cuda")
    attention_mask = encoding['attention_mask'].to("cuda")
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    #expl= explanations.generate_LRP_last_layer(input_ids, attention_mask)
    expl = (expl - expl.min()) / (expl.max() - expl.min())

    # get the model classification
    output = torch.nn.functional.softmax(model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
    classification = output.argmax(dim=-1).item()
    class_name = classifications[classification]
    print(sentences[i], "---is predicted: --", class_name)
    if class_name == "toxic":
      expl *= (-1)
    if class_name == "toxic":
        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
        print([(tokens[i], expl[i].item()) for i in range(len(tokens))])

# %%

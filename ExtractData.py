import os
import pandas as pd
import random
# https://deepai.org/dataset/conll-2003-english
#categories: LOC, ORG, PER, MISC

# The chunk tags and the named entity tags have the format I-TYPE
# which means that the word is inside a phrase of type TYPE.
# Only if two phrases of the same type immediately follow each other,
# the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase.
def split_text_label(filename):
  f = open(filename)
  split_labeled_text = []
  sentence = []
  for line in f:
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
       if len(sentence) > 0:
         split_labeled_text.append(sentence)
         sentence = []
       continue
    splits = line.split(' ')
    sentence.append([splits[0],splits[-1].rstrip("\n")])
  if len(sentence) > 0:
    split_labeled_text.append(sentence)
    sentence = []
  return split_labeled_text

print("full train, valid, test set")
split_train = split_text_label(os.path.join("data", "train.txt"))
print(len(split_train))

split_valid = split_text_label(os.path.join("data", "valid.txt"))
print(len(split_valid))

split_test = split_text_label(os.path.join("data", "test.txt"))
print(len(split_test))

def get_NER_position_positive_data(dataset):
  training_data = []
  masked_count = []
  for data in dataset:
    complete_sentences = []
    count = 0
    count_frq = []
    for ind, word in enumerate(data):
      nonNer = 'O'
      if word[1] != nonNer :
        complete_sentences.append("<MASK>")
        count += 1
      else:
        if (count != 0):
          count_frq.append([count,ind-count])
        count = 0
        complete_sentences.append(word[0])
    if (count != 0):
        count_frq.append([count,ind-count+1])
    training_data.append(complete_sentences)
    masked_count.append(count_frq)
  return training_data,  masked_count


def get_NER_masked_positive_data(dataset, nerPosition):
  training_data = []
  masked_count = []
  for ind_d, data in enumerate(dataset):
    nerSent = nerPosition[ind_d]
    for ner in nerSent:
      complete_sentences = []
      numberOfNer = ner[0]
      wordPosition = ner[1]
      for ind, word in enumerate(data):
        if (ind >= wordPosition and ind < (wordPosition + numberOfNer)):
          complete_sentences.append("<MASK>")
        else:
          complete_sentences.append(word[0])
      training_data.append([complete_sentences,1])
  return training_data


def get_NER_masked_negative_data(dataset, nerPosition):
  training_data = []
  nonNer = 'O'
  for ind_d, data in enumerate(dataset):
    nerSent = nerPosition[ind_d]
    for ner in nerSent:
      complete_sentences = []
      numberOfNer = ner[0]
      wordPosition = ner[1] + numberOfNer
      for ind, word in enumerate(data):
        if (word[1] == nonNer and ind >= wordPosition and ind < (wordPosition + numberOfNer)):
          complete_sentences.append("<MASK>")
        else:
          complete_sentences.append(word[0])
      training_data.append([complete_sentences,0])
  return training_data

def get_NER_masked_alternative_negative_data(dataset, nerPosition):
  nonNer = 'O'
  training_data = []
  for ind_d, data in enumerate(dataset):
    nerSent = nerPosition[ind_d]
    for ind_n, ner in enumerate(nerSent):
      complete_sentences = []
      numberOfNer = ner[0]
      wordPosition = ner[1]
      startPosition = wordPosition + numberOfNer
      #nextNer = startPosition
      nextNer = len(data) - 1
      if ind_n < (len(nerSent) - 1):
        nextNer = nerSent[ind_n + 1][1]

      if (startPosition <= (nextNer - numberOfNer)):
        endPosition =  nextNer - numberOfNer
      else:
        endPosition = startPosition

      randomStartPoint = random.randint(startPosition, endPosition)

      for ind, word in enumerate(data):
        if (word[1] == nonNer and ind >= randomStartPoint and ind < (randomStartPoint + numberOfNer)):
          complete_sentences.append("<MASK>")
        else:
          complete_sentences.append(word[0])
      training_data.append([complete_sentences,0])
  return training_data

def convert_list_to_sentence(sent_list):
  sentences = []
  for line in sent_list:
    joined = ' '.join(line[0])
    sentences.append([joined,line[1]])
  return sentences

def get_dataset(data):
  masked_training_data_pos, masked_count = get_NER_position_positive_data(data)
  masked_training_data_positive = get_NER_masked_positive_data(data, masked_count)
  masked_training_data_pos_joined = convert_list_to_sentence(masked_training_data_positive)

  print("original sentence")
  print(data[0])
  print(data[1])
  print(data[2])
  print(data[3])
  # masked_training_data_neg = get_NER_masked_negative_data(data, masked_count)
  # masked_training_data_neg_joined = convert_list_to_sentence(masked_training_data_neg)
  print(masked_training_data_pos_joined[0])
  print(masked_training_data_pos_joined[1])
  print(masked_training_data_pos_joined[2])
  print(masked_training_data_pos_joined[3])
  print(masked_training_data_pos_joined[4])

  masked_training_data_neg = get_NER_masked_alternative_negative_data(data, masked_count)
  masked_training_data_neg_joined = convert_list_to_sentence(masked_training_data_neg)

  print(masked_training_data_neg_joined[0])
  print(masked_training_data_neg_joined[1])
  print(masked_training_data_neg_joined[2])
  print(masked_training_data_neg_joined[3])
  print(masked_training_data_neg_joined[4])

  masked_data_df = pd.DataFrame(masked_training_data_pos_joined + masked_training_data_neg_joined, columns=['text', 'label'])
  masked_data_df = masked_data_df.sample(frac=1)

  return masked_data_df.reset_index(drop=True)


print("small training data")
#training_data = get_dataset(split_train)
small_training_data = get_dataset(split_train[0:1500])

print(small_training_data.count())

print("small valid data")
# valid_data = get_dataset(split_valid)
small_valid_data = get_dataset(split_valid[0:500])
# #print(valid_data.count())
print(small_valid_data.count())
#
print("small test data")
# test_data = get_dataset(split_test)
small_test_data = get_dataset(split_test[0:500])
# print(test_data.count())
print(small_test_data.count())
#
# training_valid_data = pd.concat([training_data, valid_data],ignore_index = True)
# print(training_valid_data.count())
#
# training_valid_data.to_pickle("data/masked_training_valid_data")
# test_data.to_pickle("data/masked_test_data")
# training_data.to_pickle("data/masked_training_data")
# valid_data.to_pickle("data/masked_valid_data")
#
small_test_data.to_csv('data/masked_small_test_data.csv', sep='\t')
small_training_data.to_csv('data/masked_small_training_data.csv', sep='\t')
small_valid_data.to_csv('data/masked_small_valid_data.csv', sep='\t')

small_test_data.to_pickle("data/masked_small_test_data")
small_training_data.to_pickle("data/masked_small_training_data")
small_valid_data.to_pickle("data/masked_small_valid_data")


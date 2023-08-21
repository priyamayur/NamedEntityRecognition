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


def get_NER_masked_postive_data(dataset, nerPosition, classLabel, classValue):
  training_data = []
  classLabel = classLabel
  for ind_d, data in enumerate(dataset):
    nerSent = nerPosition[ind_d]
    for ner in nerSent:
      complete_sentences = []
      numberOfNer = ner[0]
      wordPosition = ner[1]
      maskFlag = 0
      for ind, word in enumerate(data):
        wordLabel = word[1]
        if (classLabel in wordLabel and ind >= wordPosition and ind < (wordPosition + numberOfNer)):
          complete_sentences.append("<MASK>")
          maskFlag = 1
        else:
          complete_sentences.append(word[0])
      if maskFlag ==1 :
        training_data.append([complete_sentences, classValue])
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
      maskFlag = 0
      for ind, word in enumerate(data):
        if (word[1] == nonNer and ind >= randomStartPoint and ind < (randomStartPoint + numberOfNer)):
          complete_sentences.append("<MASK>")
          maskFlag = 1
        else:
          complete_sentences.append(word[0])
      if maskFlag == 1:
        training_data.append([complete_sentences,0])
  return training_data

def convert_list_to_sentence(sent_list):
  sentences = []
  for line in sent_list:
    joined = ' '.join(line[0])
    sentences.append([joined,line[1]])
  return sentences

def get_dataset(data, dataClassLimit):
  masked_training_data_pos, masked_count = get_NER_position_positive_data(data)
  masked_training_data_positive_org = get_NER_masked_postive_data(data, masked_count, "ORG", 1)
  masked_training_data_pos_org_joined = convert_list_to_sentence(masked_training_data_positive_org)
  masked_training_data_pos_org_joined = masked_training_data_pos_org_joined[0:dataClassLimit]

  print("original sentence")
  print(data[0])
  print(data[1])
  print(data[2])
  print(data[3])
  print("class ORG")
  print(masked_training_data_pos_org_joined[0])
  print(masked_training_data_pos_org_joined[1])
  print(masked_training_data_pos_org_joined[2])
  print(masked_training_data_pos_org_joined[3])
  print(masked_training_data_pos_org_joined[4])
  print(masked_training_data_pos_org_joined[5])
  print(masked_training_data_pos_org_joined[6])
  print(masked_training_data_pos_org_joined[7])
  # print(len(masked_training_data_pos_B_joined))

  masked_training_data_pos, masked_count = get_NER_position_positive_data(data)
  masked_training_data_positive_misc = get_NER_masked_postive_data(data, masked_count, "MISC", 2)
  masked_training_data_pos_misc_joined = convert_list_to_sentence(masked_training_data_positive_misc)
  masked_training_data_pos_misc_joined = masked_training_data_pos_misc_joined[0:dataClassLimit]

  print("class misc")
  # print(len(masked_training_data_pos_I_joined))
  print(masked_training_data_pos_misc_joined[0])
  print(masked_training_data_pos_misc_joined[1])

  masked_training_data_pos, masked_count = get_NER_position_positive_data(data)
  masked_training_data_positive_loc = get_NER_masked_postive_data(data, masked_count, "LOC", 3)
  masked_training_data_pos_loc_joined = convert_list_to_sentence(masked_training_data_positive_loc)
  masked_training_data_pos_loc_joined = masked_training_data_pos_loc_joined[0:dataClassLimit]

  print("class loc")
  # print(len(masked_training_data_pos_I_joined))
  print(masked_training_data_pos_loc_joined[0])
  print(masked_training_data_pos_loc_joined[1])

  masked_training_data_pos, masked_count = get_NER_position_positive_data(data)
  masked_training_data_positive_per = get_NER_masked_postive_data(data, masked_count, "PER", 4)
  masked_training_data_pos_per_joined = convert_list_to_sentence(masked_training_data_positive_per)
  masked_training_data_pos_per_joined = masked_training_data_pos_per_joined[0:dataClassLimit]

  print("class per")
  # print(len(masked_training_data_pos_I_joined))
  print(masked_training_data_pos_per_joined[0])
  print(masked_training_data_pos_per_joined[1])

  masked_training_data_neg = get_NER_masked_alternative_negative_data(data, masked_count)
  masked_training_data_neg_joined = convert_list_to_sentence(masked_training_data_neg)
  masked_training_data_neg_joined = masked_training_data_neg_joined[0:dataClassLimit]

  print("class 0")
  # print(len(masked_training_data_neg_joined))
  print(masked_training_data_neg_joined[0])
  print(masked_training_data_neg_joined[1])
  print(masked_training_data_neg_joined[2])
  print(masked_training_data_neg_joined[3])
  print(masked_training_data_neg_joined[4])
  print(masked_training_data_neg_joined[5])
  print(masked_training_data_neg_joined[6])


  masked_data_df = pd.DataFrame(masked_training_data_pos_org_joined + masked_training_data_pos_misc_joined +
                                masked_training_data_pos_misc_joined + masked_training_data_pos_per_joined +
                                masked_training_data_neg_joined, columns=['text', 'label'])
  masked_data_df = masked_data_df.sample(frac=1)

  return masked_data_df.reset_index(drop=True)


print("small training data")
small_training_data = get_dataset(split_train[0:2500], 1500)
# sampleText = list(small_training_data['text'][0:5])
# sampleLabel = small_training_data['label'][0:5]
# print(sampleText)
print(small_training_data.count())


print("small valid data")
small_valid_data = get_dataset(split_valid[0:1200], 500)
print(small_valid_data.count())


print("small test data")
small_test_data = get_dataset(split_test[0:1200],500)
print(small_test_data.count())

small_test_data.to_csv('data/experiment3_test_data.csv', sep='\t')
small_training_data.to_csv('data/experiment3_training_data.csv', sep='\t')
small_valid_data.to_csv('data/experiment3_valid_data.csv', sep='\t')


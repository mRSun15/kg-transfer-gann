import pickle
import json
import os
from gensim.models import Word2Vec
import gensim
import numpy as np


class MySentences(object):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        #         for fname in os.listdir(self.dirname):
        #             for line in open(os.path.join(self.dirname, fname)):
        #                 yield line.split()
        for filename in self.files:
            for line in open(filename):
                temp_sentence = line.split('\t')[-1]
                yield temp_sentence.split()
def GetEntityPos(entity1, entity2, sentence):
    pos_1 = 0
    pos_2 = 0
    for i in range(len(sentence)):
        if sentence[i] == entity1:
            pos_1 = i
        if sentence[i] == entity2:
            pos_2 = i
    pos_first = min(pos_1, pos_2)
    pos_second = pos_1 + pos_2 - pos_first
    return [pos_first, pos_second]
def LoadFile(filename, max_length, vec_size, pos_size, word_2_vec_model, relation_set):

    data_embedding = np.zeros((324704, max_length ,vec_size+pos_size))
    relation_label = []
    print("Process file:",filename)
    count = 0
    for line in open(filename):
        if (count%10000) == 0:
            print("Now the count is: ",count)

        split_sentce = line.split('\t')
        relation_id = split_sentce[2]
        flag = 0
        for (key, relation) in relation_set.items():
            if relation[1] == relation_id:
                relation_id = relation[0]
                flag = 1
                break
        if flag == 0:
            continue
        relation_label.append(relation_id)
        entity_1 = split_sentce[0]
        entity_2 = split_sentce[1]
        words = split_sentce[-1].split()
        word_embedding = np.zeros((max_length, vec_size))
        pos_embedding = np.zeros((max_length, pos_size))
        (entity_pos_1, entity_pos_2) = GetEntityPos(entity_1, entity_2, words)
        pos_embedding[entity_pos_1] = np.ones(pos_size)
        pos_embedding[entity_pos_2] = np.ones(pos_size)
        for i in range(len(words)):
            word_embedding[i] = word_2_vec_model[words[i]]
        input_embedding = np.concatenate((word_embedding, pos_embedding), axis = 1)
        data_embedding[count] = input_embedding
        count += 1
    print("The total count is : ",count)
    return (data_embedding[0:len(relation_label)],relation_label)

data = pickle.load(open("../unknow_data/relation_dict.pkl", 'rb'))
wiki_relation_map = json.load(open("../WikipediaWikidataDistantSupervisionAnnotations.v1.0/en.json", 'r'))

is_training = False

wiki_data_path = '../data/wiki_data/'
files = os.listdir(wiki_data_path)
file_name = 'training'
wiki_file = '../data/wiki_data/filterer-'+file_name+'.txt'
wiki_files = ['../data/wiki_data/filterer-held-out.txt','../data/wiki_data/filterer-training.txt','../data/wiki_data/filterer-validation.txt']
if is_training:

    # temp_file = open(wiki_file, 'r')f
    # temp_file_lines = [line for line in temp_file.readlines()]
    sentences = MySentences(wiki_files)
    model = gensim.models.Word2Vec(sentences, min_count=1, size=50)
    model.save('../Preprocess_model_pkl/wiki_vec_model.pkl')
else:
    model = gensim.models.Word2Vec.load('../Preprocess_model_pkl/wiki_vec_model.pkl')
relation_set = pickle.load(open('../data/common_relations.pkl', 'rb'))
data, label = LoadFile(wiki_file, 100, vec_size=50, pos_size=5, word_2_vec_model=model, relation_set=relation_set)
np.save('../data/processed_data/wiki_'+file_name+'_data', data)
np.save('../data/processed_data/wiki_'+file_name+'_label', np.array(label))
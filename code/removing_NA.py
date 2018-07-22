import numpy as np

wiki_training_data = np.load('../data/wiki_training_data.npy')
wiki_training_label = np.load('../data/wiki_training_label.npy')
wiki_held_out_data = np.load('../data/wiki_held-out_data.npy')
wiki_held_out_label = np.load('../data/wiki_held-out_label.npy')
wiki_validation_data = np.load('../data/wiki_validation_data.npy')
wiki_validation_label = np.load('../data/wiki_validation_label.npy')
nyt_train_data = np.load('../data/nyt_train_data.npy')
nyt_train_label = np.load('../data/nyt_train_label.npy')
nyt_test_data = np.load('../data/nyt_test_data.npy')
nyt_test_label = np.load('../data/nyt_test_label.npy')


#remove the NA relation, NA's id is 3
relation_id = 3
new_wiki_training_data = wiki_training_data[wiki_training_label != relation_id]
new_wiki_training_label = wiki_training_label[wiki_training_label != relation_id]
new_wiki_held_out_data = wiki_held_out_data[wiki_held_out_label != relation_id]
new_wiki_held_out_label = wiki_held_out_label[wiki_held_out_label != relation_id]
new_wiki_validation_data = wiki_validation_data[wiki_validation_label != relation_id]
new_wiki_validation_label = wiki_validation_label[wiki_validation_label != relation_id]
new_nyt_train_data = nyt_train_data[nyt_train_label != relation_id]
new_nyt_train_label = nyt_train_label[nyt_train_label != relation_id]
new_nyt_test_data = nyt_train_data[nyt_test_label != relation_id]
new_nyt_test_label = nyt_train_label[nyt_test_label != relation_id]

#save
np.save('../data/wiki_train_data_.npy', new_wiki_training_data)
np.save('../data/wiki_train_label_.npy', new_wiki_training_label)
np.save('../data/wiki_held_data_.npy', new_wiki_held_out_data)
np.save('../data/wiki_held_label_.npy', new_wiki_held_out_label)
np.save('../data/wiki_validat_data_.npy', new_wiki_validation_data)
np.save('../data/wiki_validat_label_.npy', new_wiki_validation_label)
np.save('../data/nyt_train_data_.npy', new_nyt_train_data)
np.save('../data/nyt_train_label_.npy', new_nyt_train_label)
np.save('../data/nyt_test_data_.npy', new_nyt_test_data)
np.save('../data/nyt_test_label_.npy', new_nyt_test_label)
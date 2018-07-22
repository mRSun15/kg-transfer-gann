import numpy as np

new_wiki_training_label = np.load('../data/wiki_train_label_.npy')
new_wiki_held_out_label = np.load('../data/wiki_held_label_.npy')
new_wiki_validation_label = np.load('../data/wiki_validat_label_.npy')
new_nyt_train_label = np.load('../data/nyt_train_label_.npy')
new_nyt_test_label = np.load('../data/nyt_test_label_.npy')

new_wiki_validation_label[new_wiki_validation_label == 8] = 3
new_wiki_held_out_label[new_wiki_held_out_label == 8] = 3
new_wiki_training_label[new_wiki_training_label == 8] = 3
new_nyt_test_label[new_nyt_test_label == 8] = 3
new_nyt_train_label[new_nyt_train_label == 8] = 3


np.save('../data/wiki_train_label_.npy', new_wiki_training_label)
np.save('../data/wiki_held_label_.npy', new_wiki_held_out_label)
np.save('../data/wiki_validat_label_.npy', new_wiki_validation_label)
np.save('../data/nyt_train_label_.npy', new_nyt_train_label)
np.save('../data/nyt_test_label_.npy', new_nyt_test_label)
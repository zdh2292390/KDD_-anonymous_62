import os
import collections
import pandas as pd

def read_data(FILE_DIR):
    dirs = os.listdir(FILE_DIR)
    os.chdir(FILE_DIR)
    AllPhrase = list()
    for file in dirs:
        path = FILE_DIR + '/' + file
        dirs = os.listdir(path)
        for file1 in dirs:
            if 'multi' in file1:
                df = pd.read_csv(path + '/' + file1, sep='\t', encoding='utf8', header=None)
                df.columns = ['score', 'phrase']
                df['Product'] = [file] * len(df)
                AllPhrase.append(df)
    return pd.concat(AllPhrase, ignore_index=True, sort=False)

def sampling(methods, AllPhrase, total_num, score_threshold, file):
    F = file + '/' + 'product_phrase'
    if not os.path.exists(F):
        os.mkdir(F)

    if methods == 'random':
        Phrases = list()
        for p in AllPhrase['Product'].unique():
            p_num = int(total_num / len(AllPhrase) * len(AllPhrase[AllPhrase['Product'] == p]))
            sample_p = AllPhrase[(AllPhrase['Product'] == p) & (AllPhrase['score'] >= score_threshold)]\
                .sample(n=p_num, weights='score', random_state=1).sort_values(by=['score'], ascending = False).reset_index()
            Phrases.append(sample_p)
        Phrases = pd.concat(Phrases, ignore_index=True, sort=False)
        # print(Phrases)
        # Phrases.to_pickle(F + '/all_phrases_random')
    elif methods == 'rank':
        Phrases = list()
        for p in AllPhrase['Product'].unique():
            p_num = int(total_num / len(AllPhrase) * len(AllPhrase[AllPhrase['Product'] == p]))
            sample_p = AllPhrase[(AllPhrase['Product'] == p) & (AllPhrase['score'] >= score_threshold)]\
                .sort_values(by=['score'], ascending=False)[:p_num].reset_index()
            Phrases.append(sample_p)
        Phrases = pd.concat(Phrases, ignore_index=True, sort=False)
        # print(Phrases)
        # Phrases.to_pickle(F + '/all_phrases_rank')
    return Phrases

def get_phrase_table():
    FILE_DIR = './phrase_pool'
    total_num = 100000
    score_threshold = 0.8
    methods = 'rank' # 'rank' or 'random'
    AllPhrase = read_data(FILE_DIR)

    return sampling(methods, AllPhrase, total_num, score_threshold, FILE_DIR)


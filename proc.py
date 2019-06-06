import os
import json
from tqdm import tqdm
import pickle
import numpy as np
import spacy
from config import Config

def tokenize(sentents):
    sents = nlp(sentents)
    return [token.text for token in sents]


def get_span(tokens, context):
    current = 0
    span = []
    for token in tokens:
        current = context.find(token, current)
        span.append((current, current + len(token)))
        current += len(token)
    return span


def get_data(file):
    with open(file) as f:
        raw_data = json.load(f)

    total = 0
    raw_data = raw_data['data']
    examples = []
    examples_eval = {}
    for data in tqdm(raw_data):
        for para in data['paragraphs']:
            context = para['context'].replace('"', "'").replace('``', '"')
            context_tokens = tokenize(context)
            context_span = get_span(context_tokens, context)
            context_chars = [list(token) for token in context_tokens]
            for qa in para['qas']:
                total += 1
                question = qa['question'].replace('"', "'").replace('``', '"')
                question_tokens = tokenize(question)
                question_chars = [list(token) for token in question_tokens]
                answer_texts = []
                for ans in qa['answers']:
                    ans_text = ans['text']
                    answer_texts.append(ans_text)
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(ans_text)
                    ans_span = []
                    for idx, span in enumerate(context_span):
                        if not (ans_start >= span[1] or ans_end <= span[0]):
                            ans_span.append(idx)
                    y1s = ans_span[0]
                    y2s = ans_span[-1]
                    example = {'context_tokens': context_tokens, 'context_chars': context_chars,'question_tokens':question_tokens,
                               'question_chars': question_chars,'y1s': y1s, 'y2s': y2s, 'uuid': total}
                    examples.append(example)
                    examples_eval[str(total)] = {'context': context, 'question': question,
                                                 'answer': answer_texts, 'context_span': context_span}
    return (examples, examples_eval)


def get_embedding(emb_file):
    embedding_dict = {}
    with open(emb_file) as f:
        for line in tqdm(f):
            array = line.split(' ')
            word = array[0]
            vector = list(map(float, array[1: ]))
            embedding_dict[word] = vector

    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx['<pad>'] = 0
    token2idx['<unk>'] = 1
    embedding_dict['<pad>'] = [0. for _ in range(300)]
    embedding_dict['<unk>'] = [np.random.normal(scale=0.1) for _ in range(300)]
    idx2embed = {idx: embedding_dict[token] for token, idx in token2idx.items()}
    embedding_mat = [idx2embed[idx] for idx in range(len(idx2embed))]
    return embedding_mat, token2idx


if __name__ == '__main__':
    data_path = '../data/squad/'
    embed_path = '../data/glove/'
    nlp = spacy.blank('en')
    config = Config()
    data_dir = './pre_data'

    # print('Begin converting raw data....')
    # data_train = get_data(os.path.join(data_path, 'train-v2.0.json'))
    # print('Done!!')
    # print('Saving raw data to %s' % data_dir)
    # pickle.dump(data_train, open(os.path.join(data_dir, 'data_train_pre.pkl'), 'wb'))
    # print('Done!!!')

    print('Begin converting raw data....')
    data_dev = get_data(os.path.join(data_path, 'dev-v2.0.json'))
    print('Done!!')
    print('Saving raw data to %s' % data_dir)
    pickle.dump(data_dev, open(os.path.join(data_dir, 'data_dev_pre.pkl'), 'wb'))
    print('Done!!!')
    #
    # print('Begin converting embedding...')
    # embed_file = os.path.join(embed_path, 'glove.6B.300d.txt')
    # print(embed_file)
    # embedding = get_embedding(embed_file)
    # pickle.dump(embedding, open(os.path.join(data_dir, 'embed_pre.pkl'), 'wb'))
    # print('Done!!!')

    # print('Get SQuAD Dataset')
    # train_dataset = SQuADData(os.path.join(data_dir, 'train_pre.pkl'))
    # dev_dataset = SQuADData(os.path.join(data_dir, 'dev_pre.pkl'))
    # pickle.dump(train_dataset, open(os.path.join(data_dir, 'train_dataset.pkl'), 'wb'))
    # pickle.dump(dev_dataset, open(os.path.join(data_dir, 'dev_dataset.pkl'), 'wb'))
    # print('Done!')

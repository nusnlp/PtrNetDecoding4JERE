import sys
import os
import numpy as np
import random

from collections import OrderedDict
import pickle
import datetime
from tqdm import tqdm
from recordclass import recordclass
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def load_word_embedding(embed_file, vocab):
    custom_print('vocab length:', len(vocab))
    embed_vocab = OrderedDict()
    rev_embed_vocab = OrderedDict()
    embed_matrix = list()

    embed_vocab['<PAD>'] = 0
    rev_embed_vocab[0] = '<PAD>'
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))

    embed_vocab['<UNK>'] = 1
    rev_embed_vocab[1] = '<UNK>'
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    embed_vocab['<SOS>'] = 2
    rev_embed_vocab[2] = '<SOS>'
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    embed_vocab['<EOS>'] = 3
    rev_embed_vocab[3] = '<EOS>'
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    word_idx = 4
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and vocab[word] >= word_min_freq:
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                embed_vocab[word] = word_idx
                rev_embed_vocab[word_idx] = word
                word_idx += 1

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_min_freq:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            rev_embed_vocab[word_idx] = word
            word_idx += 1

    custom_print('embed dictionary length:', len(embed_vocab))
    return embed_vocab, rev_embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(data, rels, vocab_file, embed_file):
    vocab = OrderedDict()
    char_v = OrderedDict()
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_v[';'] = 2
    char_v['|'] = 3
    char_idx = 4
    for d in data:
        for word in d.SrcWords:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

            for c in word:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1

    for rel in rels:
        vocab[rel] = word_min_freq

    vocab[';'] = word_min_freq
    vocab['|'] = word_min_freq

    word_v, rev_word_v, embed_matrix = load_word_embedding(embed_file, vocab)
    output = open(vocab_file, 'wb')
    pickle.dump([word_v, char_v], output)
    output.close()
    return word_v, rev_word_v, char_v, embed_matrix


def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        word_v, char_v = pickle.load(f)
    return word_v, char_v


def get_adj_mat(amat):
    K = 5
    adj_mat = np.zeros((len(amat), len(amat)), np.float32)
    for i in range(len(amat)):
        for j in range(len(amat)):
            if 0 <= amat[i][j] <= K:
                adj_mat[i][j] = 1.0 / math.pow(2, amat[i][j])
            else:
                adj_mat[i][j] = 0
    return adj_mat


def get_data(src_lines, trg_lines, adj_lines, datatype):
    samples = []
    uid = 1
    src_len = -1
    trg_len = -1
    for i in range(0, len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()
        src_words = src_line.split()

        if datatype == 1:
            tuples = trg_line.strip().split('|')
            random.shuffle(tuples)
            new_trg_line = ' | '.join(tuples)
            assert len(trg_line.split()) == len(new_trg_line.split())
            trg_line = new_trg_line

        trg_words = list()
        trg_words.append('<SOS>')
        trg_words += trg_line.split()
        trg_words.append('<EOS>')

        adj_data = json.loads(adj_lines[i])
        adj_mat = get_adj_mat(adj_data['adj_mat'])

        if datatype == 1 and (len(src_words) > max_src_len or len(trg_words) > max_trg_len + 1):
            continue
        if len(src_words) > src_len:
            src_len = len(src_words)
        if len(trg_words) > trg_len:
            trg_len = len(trg_words)
        sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_words),
                        TrgWords=trg_words, AdjMat=adj_mat)
        samples.append(sample)
        uid += 1
    print(src_len)
    print(trg_len)
    return samples


def read_data(src_file, trg_file, adj_file, datatype):
    reader = open(src_file)
    src_lines = reader.readlines()
    reader.close()

    reader = open(trg_file)
    trg_lines = reader.readlines()
    reader.close()

    reader = open(adj_file)
    adj_lines = reader.readlines()
    reader.close()

    # tot_len = 100
    # src_lines = src_lines[0:min(tot_len, len(src_lines))]
    # trg_lines = trg_lines[0:min(tot_len, len(trg_lines))]
    # adj_lines = adj_lines[0:min(tot_len, len(adj_lines))]

    data = get_data(src_lines, trg_lines, adj_lines, datatype)
    return data


def get_relations(file_name):
    rels = []
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    for line in lines:
        rels.append(line.strip())
    return rels


def get_pred_words(preds, attns, src_words):
    pred_words = []
    for i in range(0, max_trg_len):
        word_idx = preds[i]
        if word_vocab['<EOS>'] == word_idx:
            pred_words.append('<EOS>')
            break
        elif att_type != 'None' and copy_on and word_vocab['<UNK>'] == word_idx:
            word_idx = attns[i]
            pred_words.append(src_words[word_idx])
        else:
            pred_words.append(rev_word_vocab[word_idx])
    return pred_words


def get_F1(data, preds, attns):
    gt_pos = 0
    pred_pos = 0
    correct_pos = 0
    for i in range(0, len(data)):
        gt_words = data[i].TrgWords[1:]
        pred_words = get_pred_words(preds[i], attns[i], data[i].SrcWords)
        gt_pos += len(gt_words)
        pred_pos += len(pred_words)
        for j in range(0, min(len(gt_words), len(pred_words))):
            if gt_words[j] == pred_words[j]:
                correct_pos += 1
    return pred_pos, gt_pos, correct_pos


def is_full_match(triplet, triplets):
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
            return True
    return False


def is_head_match(triplet, triplets, cur_mode):
    if cur_mode == 1:
        return is_full_match(triplet, triplets)
    for t in triplets:
        if t[0].split()[-1] == triplet[0].split()[-1] \
                and t[1].split()[-1] == triplet[1].split()[-1] \
                and t[2].split()[-1] == triplet[2].split()[-1]:
            return True
    return False


def cal_f1(ref_lines, pred_lines, rel_lines, cur_mode):
    rels = []
    for line in rel_lines:
        rels.append(line.strip())
    gt_pos = 0
    pred_pos = 0
    correct = 0
    total_pred_triple = 0
    total_gt_triple = 0
    none_cnt = 0
    same_cnt = 0
    dup = 0
    for i in range(0, min(len(ref_lines), len(pred_lines))):
        ref_line = ref_lines[i].strip()
        ref_triplets = []
        if ref_line != 'NIL':
            for t in ref_line.split('|'):
                parts = t.split(';')
                triplet = (parts[0].strip(), parts[1].strip(), parts[2].strip())
                total_gt_triple += 1
                if not is_full_match(triplet, ref_triplets):
                    ref_triplets.append(triplet)
            gt_pos += len(ref_triplets)

        pred_line = pred_lines[i].strip()
        if pred_line == 'NIL' or pred_line == '':
            continue
        pred_triplets = []
        for t in pred_line.split('|'):
            parts = t.split(';')
            if len(parts) != 3:
                continue
            em1 = parts[0].strip()
            em2 = parts[1].strip()
            rel = parts[2].strip()

            if len(em1) == 0 or len(em2) == 0 or len(rel) == 0:
                continue

            if em1 == em2:
                same_cnt += 1
                continue

            if rel not in rels:
                continue

            if rel == 'None' or em1 == 'None' or em2 == 'None':
                none_cnt += 1
                continue

            triplet = (em1, em2, rel)
            total_pred_triple += 1
            if not is_full_match(triplet, pred_triplets):
                pred_triplets.append(triplet)
            else:
                dup += 1

        pred_pos += len(pred_triplets)
        for gt_triplet in ref_triplets:
            if is_head_match(gt_triplet, pred_triplets, cur_mode):
                correct += 1

    # print(pred_pos, '\t', gt_pos, '\t', correct, '\t', total_pred_triple, '\t', total_gt_triple, '\t', none_cnt)
    p = float(correct / (pred_pos + 1e-08))
    r = float(correct / (gt_pos + 1e-08))
    f1 = 2 * p * r / (p + r + 1e-08)
    p = round(p, 3)
    r = round(r, 3)
    f1 = round(f1, 3)
    # res = [p, r, f1]
    # print(res)
    # print(r)
    # print(f1)
    # print(same_cnt)
    # print(dup)
    return p, r, f1


def write_test_res(data, preds, attns, outfile):
    writer = open(outfile, 'w')
    for i in range(0, len(data)):
        pred_words = get_pred_words(preds[i], attns[i], data[i].SrcWords)[:-1]
        writer.write(' '.join(pred_words) + '\n')
    writer.close()


def shuffle_data(data):
    custom_print(len(data))
    data.sort(key=lambda x: x.SrcLen)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_max_len(sample_batch):
    src_max_len = len(sample_batch[0].SrcWords)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].SrcWords) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)

    trg_max_len = len(sample_batch[0].TrgWords)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].TrgWords) > trg_max_len:
            trg_max_len = len(sample_batch[idx].TrgWords)

    return src_max_len, trg_max_len


def get_words_index_seq(words, max_len):
    seq = list()
    for word in words:
        if word in word_vocab:
            seq.append(word_vocab[word])
        else:
            seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        seq.append(word_vocab['<PAD>'])
    return seq


def get_target_words_index_seq(words, max_len):
    seq = list()
    for word in words:
        if word in word_vocab:
            seq.append(word_vocab[word])
        else:
            seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        seq.append(word_vocab['<EOS>'])
    return seq


def get_padded_mask(cur_len, max_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_target_vocab_mask(src_words):
    mask = []
    for i in range(0, len(word_vocab)):
        mask.append(1)
    for word in src_words:
        if word in word_vocab:
            mask[word_vocab[word]] = 0
    for rel in relations:
        mask[word_vocab[rel]] = 0

    mask[word_vocab['<UNK>']] = 0
    mask[word_vocab['<EOS>']] = 0
    mask[word_vocab[';']] = 0
    mask[word_vocab['|']] = 0
    return mask


def get_rel_mask(trg_words, max_len):
    mask_seq = list()
    for word in trg_words:
        mask_seq.append(0)
        # if word in relations:
        #     mask_seq.append(0)
        # else:
        #     mask_seq.append(1)
    pad_len = max_len - len(trg_words)
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_char_seq(words, max_len):
    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])
    return char_seq


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    src_words_list = list()
    src_words_mask_list = list()
    src_char_seq = list()

    trg_words_list = list()
    trg_vocab_mask = list()
    adj_lst = []

    target = list()
    cnt = 0
    for sample in cur_samples:
        src_words_list.append(get_words_index_seq(sample.SrcWords, batch_src_max_len))
        src_words_mask_list.append(get_padded_mask(sample.SrcLen, batch_src_max_len))
        src_char_seq.append(get_char_seq(sample.SrcWords, batch_src_max_len))
        trg_vocab_mask.append(get_target_vocab_mask(sample.SrcWords))

        cur_masked_adj = np.zeros((batch_src_max_len, batch_src_max_len), dtype=np.float32)
        cur_masked_adj[:len(sample.SrcWords), :len(sample.SrcWords)] = sample.AdjMat
        adj_lst.append(cur_masked_adj)

        if is_training:
            padded_trg_words = get_words_index_seq(sample.TrgWords, batch_trg_max_len)
            trg_words_list.append(padded_trg_words)
            target.append(padded_trg_words[1:])
        else:
            trg_words_list.append(get_words_index_seq(['<SOS>'], 1))
        cnt += 1

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'src_chars': np.array(src_char_seq),
            'src_words_mask': np.array(src_words_mask_list),
            'adj': np.array(adj_lst),
            'trg_vocab_mask': np.array(trg_vocab_mask),
            'trg_words': np.array(trg_words_list, dtype=np.int32),
            'target': np.array(target)}


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds


class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim):
        self.drop_rate = drop_rate
        super(GCN, self).__init__()
        self.gcn_num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.gcn_num_layers):
            self.gcn_layers.append(nn.Linear(in_dim, out_dim))
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, gcn_input, adj):
        denom = torch.sum(adj, 2).unsqueeze(2) + 1
        for i in range(self.gcn_num_layers):
            Ax = torch.bmm(adj, gcn_input)
            AxW = self.gcn_layers[i](Ax)
            AxW = AxW + self.gcn_layers[i](gcn_input)
            AxW /= denom
            gAxW = F.relu(AxW)
            gcn_input = self.dropout(gAxW) if i < self.gcn_num_layers - 1 else gAxW
        return gcn_input


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        if enc_type == 'LSTM':
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)
        elif enc_type == 'GCN':
            self.reduce_dim = nn.Linear(self.input_dim, 2 * self.hidden_dim)
            self.gcn = GCN(gcn_num_layers, 2* self.hidden_dim, 2 * self.hidden_dim)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)
            self.gcn = GCN(gcn_num_layers, 2 * self.hidden_dim, 2 * self.hidden_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)

    def forward(self, words_input, char_seq, adj, is_training=False):
        char_embeds = self.char_embeddings(char_seq)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((words_input, char_feature), -1)
        if enc_type == 'LSTM':
            outputs, hc = self.lstm(words_input)
            outputs = self.dropout(outputs)
        elif enc_type == 'GCN':
            outputs = self.reduce_dim(words_input)
            outputs = self.gcn(outputs, adj)
            outputs = self.dropout(outputs)
        else:
            outputs, hc = self.lstm(words_input)
            outputs = self.dropout(outputs)
            outputs = self.gcn(outputs, adj)
            outputs = self.dropout(outputs)
        return outputs


def mean_over_time(x, mask):
    x.data.masked_fill_(mask.unsqueeze(2).data, 0)
    x = torch.sum(x, dim=1)
    time_steps = torch.sum(mask.eq(0), dim=1, keepdim=True).float()
    x /= time_steps
    return x


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights


class NGram_Attention(nn.Module):
    def __init__(self, input_dim, N):
        super(NGram_Attention, self).__init__()
        self.input_dim = input_dim
        self.layers = N
        self.V_layers = nn.ModuleList()
        self.W_layers = nn.ModuleList()
        for i in range(N):
            self.V_layers.append(nn.Linear(input_dim, input_dim))
            self.W_layers.append(nn.Linear(input_dim, input_dim))

    def forward(self, s_prev, enc_hs, src_mask):
        att = torch.bmm(s_prev.unsqueeze(1), self.V_layers[0](enc_hs).transpose(1, 2)).squeeze()
        att.data.masked_fill_(src_mask.data, -float('inf'))
        att = F.softmax(att, dim=-1)
        ctx = self.W_layers[0](torch.bmm(att.unsqueeze(1), enc_hs).squeeze())
        for i in range(1, self.layers):
            enc_hs_ngram = torch.nn.AvgPool1d(i+1, 1)(enc_hs.transpose(1, 2)).transpose(1, 2)
            n_mask = src_mask.unsqueeze(1).float()
            n_mask = torch.nn.AvgPool1d(i+1, 1)(n_mask).squeeze()
            n_mask[n_mask > 0] = 1
            n_mask = n_mask.byte()
            n_att = torch.bmm(s_prev.unsqueeze(1), self.V_layers[i](enc_hs_ngram).transpose(1, 2)).squeeze()
            n_att.data.masked_fill_(n_mask.data, -float('inf'))
            n_att = F.softmax(n_att, dim=-1)
            ctx += self.W_layers[i](torch.bmm(n_att.unsqueeze(1), enc_hs_ngram).squeeze())
        return ctx, att


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length
        if att_type == 'None':
            self.lstm = nn.LSTMCell(2 * self.input_dim, self.hidden_dim, self.layers)
        elif att_type == 'Unigram':
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(2 * self.input_dim, self.hidden_dim, self.layers)
        else:
            self.attention = NGram_Attention(input_dim, 3)
            self.lstm = nn.LSTMCell(3 * self.input_dim, self.hidden_dim, self.layers)

        self.dropout = nn.Dropout(self.drop_rate)
        self.ent_out = nn.Linear(self.input_dim, len(word_vocab))

    def forward(self, y_prev, h_prev, enc_hs, src_word_embeds, src_mask, is_training=False):
        src_time_steps = enc_hs.size()[1]
        if att_type == 'None':
            ctx = mean_over_time(enc_hs, src_mask)
            attn_weights = torch.zeros(src_mask.size()).cuda()
        elif att_type == 'Unigram':
            s_prev = h_prev[0]
            s_prev = s_prev.unsqueeze(1)
            s_prev = s_prev.repeat(1, src_time_steps, 1)
            ctx, attn_weights = self.attention(s_prev, enc_hs, src_mask)
        else:
            last_index = src_mask.size()[1] - torch.sum(src_mask, dim=-1).long() - 1
            last_index = last_index.unsqueeze(1).unsqueeze(1).repeat(1, 1, enc_hs.size()[-1])
            enc_last = torch.gather(enc_hs, 1, last_index).squeeze()
            ctx, attn_weights = self.attention(enc_last, src_word_embeds, src_mask)
            ctx = torch.cat((enc_last, ctx), -1)

        y_prev = y_prev.squeeze()
        s_cur = torch.cat((y_prev, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)
        output = self.ent_out(hidden)
        return output, (hidden, cell_state), attn_weights


class SeqToSeqModel(nn.Module):
    def __init__(self):
        super(SeqToSeqModel, self).__init__()
        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, drop_rate)
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), layers, True, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, layers, drop_rate, max_trg_len)

    def forward(self, src_words_seq, src_chars_seq, src_mask, trg_words_seq, trg_vocab_mask, adj, is_training=False):
        src_word_embeds = self.word_embeddings(src_words_seq)
        trg_word_embeds = self.word_embeddings(trg_words_seq)

        batch_len = src_word_embeds.size()[0]
        if is_training:
            time_steps = trg_word_embeds.size()[1] - 1
        else:
            time_steps = max_trg_len

        encoder_output = self.encoder(src_word_embeds, src_chars_seq, adj, is_training)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, word_embed_dim)))
        h0 = h0.cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, word_embed_dim)))
        c0 = c0.cuda()
        dec_hid = (h0, c0)

        if is_training:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                      src_mask, is_training)
            dec_out = dec_out.view(-1, len(word_vocab))
            dec_out = F.log_softmax(dec_out, dim=-1)
            dec_out = dec_out.unsqueeze(1)
            for t in range(1, time_steps):
                dec_inp = trg_word_embeds[:, t, :]
                cur_dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                              src_mask, is_training)
                cur_dec_out = cur_dec_out.view(-1, len(word_vocab))
                dec_out = torch.cat((dec_out, F.log_softmax(cur_dec_out, dim=-1).unsqueeze(1)), 1)
        else:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                      src_mask, is_training)
            dec_out = dec_out.view(-1, len(word_vocab))
            if copy_on:
                dec_out.data.masked_fill_(trg_vocab_mask.data, -float('inf'))
            dec_out = F.log_softmax(dec_out, dim=-1)
            topv, topi = dec_out.topk(1)
            dec_out_v, dec_out_i = dec_out.topk(1)
            dec_attn_v, dec_attn_i = dec_attn.topk(1)

            for t in range(1, time_steps):
                dec_inp = self.word_embeddings(topi.squeeze().detach())
                cur_dec_out, dec_hid, cur_dec_attn = self.decoder(dec_inp, dec_hid, encoder_output, src_word_embeds,
                                                                  src_mask, is_training)
                cur_dec_out = cur_dec_out.view(-1, len(word_vocab))
                if copy_on:
                    cur_dec_out.data.masked_fill_(trg_vocab_mask.data, -float('inf'))
                cur_dec_out = F.log_softmax(cur_dec_out, dim=-1)
                topv, topi = cur_dec_out.topk(1)
                cur_dec_out_v, cur_dec_out_i = cur_dec_out.topk(1)
                dec_out_i = torch.cat((dec_out_i, cur_dec_out_i), 1)
                cur_dec_attn_v, cur_dec_attn_i = cur_dec_attn.topk(1)
                dec_attn_i = torch.cat((dec_attn_i, cur_dec_attn_i), 1)

        if is_training:
            dec_out = dec_out.view(-1, len(word_vocab))
            return dec_out
        else:
            return dec_out_i, dec_attn_i


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def get_model(model_id):
    if model_id == 1:
        return SeqToSeqModel()


def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    preds = list()
    attns = list()
    model.eval()
    set_random_seeds(random_seed)
    start_time = datetime.datetime.now()
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('uint8'))
        trg_vocab_mask = torch.from_numpy(cur_samples_input['trg_vocab_mask'].astype('uint8'))
        trg_words_seq = torch.from_numpy(cur_samples_input['trg_words'].astype('long'))
        adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))
        src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))

        if torch.cuda.is_available():
            src_words_seq = src_words_seq.cuda()
            src_words_mask = src_words_mask.cuda()
            trg_vocab_mask = trg_vocab_mask.cuda()
            trg_words_seq = trg_words_seq.cuda()
            adj = adj.cuda()
            src_chars_seq = src_chars_seq.cuda()

        src_words_seq = autograd.Variable(src_words_seq)
        src_words_mask = autograd.Variable(src_words_mask)
        trg_vocab_mask = autograd.Variable(trg_vocab_mask)
        adj = autograd.Variable(adj)
        src_chars_seq = autograd.Variable(src_chars_seq)

        trg_words_seq = autograd.Variable(trg_words_seq)
        with torch.no_grad():
            if model_id == 1:
                outputs = model(src_words_seq, src_chars_seq, src_words_mask, trg_words_seq, trg_vocab_mask, adj,
                                False)

        preds += list(outputs[0].data.cpu().numpy())
        attns += list(outputs[1].data.cpu().numpy())
        model.zero_grad()
    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return preds, attns


def train_model(model_id, train_samples, dev_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1
        set_random_seeds(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)

            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('uint8'))
            trg_vocab_mask = torch.from_numpy(cur_samples_input['trg_vocab_mask'].astype('uint8'))
            trg_words_seq = torch.from_numpy(cur_samples_input['trg_words'].astype('long'))
            adj = torch.from_numpy(cur_samples_input['adj'].astype('float32'))
            src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))

            target = torch.from_numpy(cur_samples_input['target'].astype('long'))

            if torch.cuda.is_available():
                src_words_seq = src_words_seq.cuda()
                src_words_mask = src_words_mask.cuda()
                trg_vocab_mask = trg_vocab_mask.cuda()
                trg_words_seq = trg_words_seq.cuda()
                adj = adj.cuda()
                src_chars_seq = src_chars_seq.cuda()

                target = target.cuda()

            src_words_seq = autograd.Variable(src_words_seq)
            src_words_mask = autograd.Variable(src_words_mask)
            trg_vocab_mask = autograd.Variable(trg_vocab_mask)
            trg_words_seq = autograd.Variable(trg_words_seq)
            adj = autograd.Variable(adj)
            src_chars_seq = autograd.Variable(src_chars_seq)

            target = autograd.Variable(target)

            if model_id == 1:
                outputs = model(src_words_seq, src_chars_seq, src_words_mask, trg_words_seq, trg_vocab_mask, adj, True)

            target = target.view(-1, 1).squeeze()
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            if (batch_idx + 1) % update_freq == 0:
                optimizer.step()
                model.zero_grad()

            train_loss_val += loss.item()

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds, dev_attns = predict(dev_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds, dev_attns)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', dev_acc)

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = int(sys.argv[2])
    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)

    src_data_folder = sys.argv[3]
    trg_data_folder = sys.argv[4]
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    job_mode = sys.argv[5]
    # run = int(sys.argv[5])
    batch_size = 32
    num_epoch = 30
    max_src_len = 100
    max_trg_len = 50
    embedding_file = os.path.join(src_data_folder, 'w2v.txt')
    update_freq = 1
    enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]
    att_type = ['None', 'Unigram', 'N-Gram-Enc'][1]

    copy_on = True

    gcn_num_layers = 3
    word_embed_dim = 300
    word_min_freq = 2
    char_embed_dim = 50
    char_feature_size = 50
    conv_filter_size = 3
    max_word_len = 10

    enc_inp_size = word_embed_dim + char_feature_size
    enc_hidden_size = word_embed_dim
    dec_inp_size = enc_hidden_size
    dec_hidden_size = dec_inp_size

    drop_rate = 0.3
    layers = 1
    early_stop_cnt = 5
    sample_cnt = 0
    Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")
    rel_file = os.path.join(src_data_folder, 'relations.txt')
    relations = get_relations(rel_file)

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print(sys.argv)
        custom_print(max_src_len, max_trg_len, drop_rate, layers)
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')
        src_train_file = os.path.join(src_data_folder, 'train.sent')
        adj_train_file = os.path.join(src_data_folder, 'train.dep')
        trg_train_file = os.path.join(src_data_folder, 'train.tup')
        train_data = read_data(src_train_file, trg_train_file, adj_train_file, 1)

        src_dev_file = os.path.join(src_data_folder, 'dev.sent')
        adj_dev_file = os.path.join(src_data_folder, 'dev.dep')
        trg_dev_file = os.path.join(src_data_folder, 'dev.tup')
        dev_data = read_data(src_dev_file, trg_dev_file, adj_dev_file, 2)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))

        custom_print("preparing vocabulary......")
        save_vocab = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, rev_word_vocab, char_vocab, word_embed_matrix = build_vocab(train_data, relations, save_vocab,
                                                                                embedding_file)

        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, char_vocab = load_vocab(vocab_file_name)

        rev_word_vocab = OrderedDict()
        for word in word_vocab:
            idx = word_vocab[word]
            rev_word_vocab[idx] = word

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        src_test_file = os.path.join(src_data_folder, 'test.sent')
        adj_test_file = os.path.join(src_data_folder, 'test.dep')
        trg_test_file = os.path.join(src_data_folder, 'test.tup')
        test_data = read_data(src_test_file, trg_test_file, adj_test_file, 3)
        custom_print('Test data size:', len(test_data))

        custom_print('seed:', random_seed)
        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        set_random_seeds(random_seed)
        test_preds, test_attns = predict(test_data, best_model, model_name)

        custom_print('Copy On')
        write_test_res(test_data, test_preds, test_attns, os.path.join(trg_data_folder, 'test.out'))

        ref_lines = open(trg_test_file).readlines()
        pred_lines = open(os.path.join(trg_data_folder, 'test.out')).readlines()
        rel_lines = open(rel_file).readlines()
        mode = 1
        custom_print('Overall F1')
        custom_print(cal_f1(ref_lines, pred_lines, rel_lines, mode))

        copy_on = False
        custom_print('Copy Off')
        set_random_seeds(random_seed)
        test_preds, test_attns = predict(test_data, best_model, model_name)
        write_test_res(test_data, test_preds, test_attns, os.path.join(trg_data_folder, 'test_without_copy.out'))

        ref_lines = open(trg_test_file).readlines()
        pred_lines = open(os.path.join(trg_data_folder, 'test_without_copy.out')).readlines()
        rel_lines = open(rel_file).readlines()
        mode = 1
        custom_print('Overall F1')
        custom_print(cal_f1(ref_lines, pred_lines, rel_lines, mode))

        logger.close()



























































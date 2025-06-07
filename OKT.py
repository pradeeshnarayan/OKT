import torch
import torch.nn as nn
import json
import numpy as np
from util import save_snapshot, load_snapshot, load_last_snapshot, open_result, Variable
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OKT(nn.Module):
        
    def load_embedding(self,filename):
        f = open(filename, encoding='utf-8')
        wcnt, emb_size = next(f).strip().split(' ')
        wcnt = int(wcnt)
        emb_size = int(emb_size)

        words = []
        embs = []
        for line in f:
            fields = line.strip().split(' ')
            #print(fields) # test
            word = fields[0]
            #print("fields 0",word)
            #print("fields 1:",fields[1:])
            emb = np.array([float(x) for x in fields[1:]])
            words.append(word)
            embs.append(emb)

        embs = np.asarray(embs)
        return wcnt, emb_size, words, embs
    
    def __init__(self, args):
        super(OKT, self).__init__()
        self.args = args
        
        wcnt, emb_size, words, embs = self.load_embedding(args.emb_file)
        self.words = words
        if args.kcnt == 0:
            print(self.args)
            know_dic = open(self.co_emb_file, 'r').read().split('\n')
            args.kcnt = len(know_dic)
        self.kcnt = args.kcnt

        
        self.seq_model = OKTSeqModel(args.topic_size, args.knowledge_hidden_size, args.kcnt,
                                         args.seq_hidden_size)

    def forward(self, co_e, ex_e, score, hidden=None, alpha=False):
        
        s, h = self.seq_model(co_e, ex_e, score, hidden)
        
        return s,h


class OKTSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, num_layers=1):
        super(OKTSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.num_layers = num_layers
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1) #stored in a continuous uniform distribution. Knowledge embedding size 25

        # Student seq rnn
        self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)


    def forward(self, o_e, ex_e, s, h, beta=None):
        if h is None:
             h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
             length = Variable(torch.FloatTensor([0.]))

        # calculate alpha weights of knowledges using dot product
        if beta is None:
            alpha = torch.mm(self.knowledge_memory, o_e.view(-1, 1)).view(-1)
            beta = nn.functional.softmax(alpha.view(1, -1), dim=-1) #student embedding size 100
            # print(beta.argmax(1))
        hkp = torch.mm(beta, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        #print("hkp:", hkp.size(), hkp)
        pred_v = torch.cat([ex_e,hkp]).view(1, -1) #pred_ size 50
        #print(pred_v.size())
        predict_score = self.score_layer(pred_v)
        
        x = ex_e
        #print("ex_e and x",ex_e.size(),ex_e,x.size(),x,s)
        x = torch.cat([x, s])
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = beta.view(-1, 1) * xk

        _, h = self.rnn(xk.unsqueeze(0), h)

        return predict_score.view(1), h


if __name__ == '__main__':
    f = open("conf.json", 'r')
    model = OKT(json.load(f))
    
    #text = Variable(torch.LongTensor(sample2))
    #print("text.float().view(1, -1)",text.float().view(1, -1))
    #print("text",text)
    score = Variable(torch.FloatTensor([1]))
    #topic_v = self.embedding(topic).mean(0, keepdim=True)
    #print(topic_v)
    #s, h = model(text, score, None)
    
    

import torch
import torch.nn as nn
import json
import numpy as np
from util import save_snapshot, load_snapshot, load_last_snapshot, open_result, Variable
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DKT(nn.Module):
    def __init__(self, args):
        super(DKT, self).__init__()
        #print(args)
        self.args = args
        self.topic_size = args.topic_size
        self.rnn = nn.GRU(self.topic_size * 2, self.topic_size, 1)
        self.score = nn.Linear(self.topic_size * 2, 1)

    def forward(self, o_e,ex_e, s, h):
        if h is None:
            h = self.default_hidden()
        otEmbeddingResize = nn.Linear(len(o_e), 100)
        exEmbeddingResize = nn.Linear(len(ex_e), 100)
        o_e=otEmbeddingResize(o_e)
        ex_e=exEmbeddingResize(ex_e)
        
        knowledgeEmbedding = nn.Linear(self.args.seq_hidden_size + self.args.knowledge_hidden_size, 100)
        outcomeExamEmbedding = torch.cat([o_e.float().view(-1), ex_e]).view(1, -1)
        knowledge = knowledgeEmbedding(outcomeExamEmbedding)

        v = knowledge.type_as(h)
        score = self.score(torch.cat([h.view(-1), v.view(-1)]))

        x = torch.cat([v.view(-1),
                       (v * (s > 0.5).type_as(v).
                        expand_as(v).type_as(v)).view(-1)])
        _, h = self.rnn(x.view(1, 1, -1), h)
        return score.view(1), h

    def default_hidden(self):
        return Variable(torch.zeros(1, 1, self.topic_size))


class OKT(nn.Module):
    emb_file = "data/emb_co.txt"
        
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
        
        wcnt, emb_size, words, embs = self.load_embedding(self.emb_file)
        self.words = words
        if args.kcnt == 0:
            print(self.args)
            know_dic = open(self.co_emb_file, 'r').read().split('\n')
            args.kcnt = len(know_dic)
        self.kcnt = args.kcnt

        # knowledge embedding module
        # convert outcome to embedding.
        #print("kcnt",self.kcnt)
        #print("args['knowledge_hidden_size']",args['knowledge_hidden_size'])
        #self.knowledge_model = KnowledgeModel(self.kcnt, args['knowledge_hidden_size'])

        # exercise embedding module
        #To convert a new question text to embedding.
        #self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'], num_layers=args['num_layers'])
        #self.topic_model.load_emb(embs)
        
        # student seq module
        self.seq_model = OKTSeqModel(args.topic_size, args.knowledge_hidden_size, args.kcnt,
                                         args.seq_hidden_size, args.k, args.score_mode)

    def forward(self, co_e, ex_e, score, hidden=None, alpha=False):
        #print("2. H",hidden)
        #exit();
        # print(knowledge.size())
        #k = self.knowledge_model(ex_e)
        # print(knowledge.size())
        #topic_h = self.topic_model.default_hidden(1)
        #topic_v, _ = self.topic_model(o_e.view(-1, 1), topic_h)
        #otEmbeddingResize = nn.Linear(len(co_e), 100).to('cuda:0')
        #exEmbeddingResize = nn.Linear(len(ex_e), 100).to('cuda:0')
        #co_e=otEmbeddingResize(co_e)
        #ex_e=exEmbeddingResize(ex_e)
        s, h = self.seq_model(co_e, ex_e, score, hidden)
        """
        if hidden is None:
            hidden = h, o_e, h
        else:
            _, o_prev, h_prev = hidden
            o_cat = torch.cat([o_prev, o_e])
            h_cat = torch.cat([h_prev, h])
            hidden = h, o_cat, h_cat

        if alpha:
            return s, hidden
        else:
            return s, hidden
        """
        return s,h


class OKTSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, k, score_mode, num_layers=1):
        super(OKTSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.score_mode = score_mode
        self.num_layers = num_layers
        # self.with_last = with_last
        #print("Setting knowledge memory ***********************")
        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1) #stored in a continuous uniform distribution. Knowledge embedding size 25
        #print(self.score_mode,self.topic_size,seq_hidden_size,num_layers)
        # Student seq rnn
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(self.topic_size * 2 + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        #print("*************",topic_size , seq_hidden_size)
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)
        #self.ft_embedding = nn.Linear(self.seq_hidden_size + self.know_emb_size, 50)
        #self.ft_embedding = nn.Linear(topic_size + seq_hidden_size, 50)
        #self.score_layer = nn.Linear(50, 1)

    def forward(self, o_e, ex_e, s, h, beta=None):
        if h is None:
             h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
             length = Variable(torch.FloatTensor([0.]))

        # calculate alpha weights of knowledges using dot product
        #print("self.knowledge_memory",self.knowledge_memory.size()) #,self.knowledge_memory)
        # print(kn.view(-1, 1))
        if beta is None:
            #print(self.knowledge_memory.size())
            #print(o_e.size())
            #print(o_e.view(-1, 1).size())
            alpha = torch.mm(self.knowledge_memory, o_e.view(-1, 1)).view(-1)
            beta = nn.functional.softmax(alpha.view(1, -1), dim=-1) #student embedding size 100
            # print(beta.argmax(1))
        #print("o_e ",o_e.view(-1, 1).size(),o_e)
        #print("beta ",beta.size(),beta)
        #print("h inside forward first:",h)
        #print(h.view(self.know_length, self.seq_hidden_size).size())
        # print(h.type())
        # predict score at time t
        #print("h", h.size(), h)
        #print("h.view", h.view(self.know_length, self.seq_hidden_size).size(), h.view(self.know_length, self.seq_hidden_size))
        hkp = torch.mm(beta, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        #print("hkp:", hkp.size(), hkp)
        pred_v = torch.cat([ex_e,hkp]).view(1, -1) #pred_ size 50
        #print(pred_v.size())
        predict_score = self.score_layer(pred_v)
        #print("pred_v",pred_v.size(),pred_v)
        #ft = torch.tanh(pred_v)
        #ft = nn.functional.softmax(pred_v.view(1, -1), dim=-1)
        #print("after_tanh",ft.size(),ft)
        #predict_score = torch.sigmoid(self.score_layer(ft))
        #ft = torch.tanh(self.ft_embedding(pred_v))
        #print("ft",ft.size(),ft)
        #predict_score = torch.sigmoid(self.score_layer(ft))
        #print("predict_score",predict_score.size(),predict_score)
        #print("actual score",s,"predicted score",predict_score)
        #print("after transformation",self.score_layer(ft),predict_score,s)
        #pred_v = nn.functional.softmax(pred_v.view(1, -1), dim=-1)
        #pred_v = torch.sigmoid(pred_v)
        #print(pred_v.size(),pred_v)
        #predict_score = torch.abs(self.score_layer(pred_v))
        
        #exit()
        # seq states update
        if self.score_mode == 'concat':
            x = ex_e
        else:
            #x = torch.cat([ex_e * (s >= 0.5).type_as(ex_e).expand_as(ex_e),
            #               ex_e * (s < 0.5).type_as(ex_e).expand_as(ex_e)])
            x = torch.cat([ex_e.view(-1),
                       (ex_e * (s > 0.5).type_as(ex_e).
                        expand_as(ex_e).type_as(ex_e)).view(-1)])
        #print("ex_e and x",ex_e.size(),ex_e,x.size(),x,s)
        x = torch.cat([x, s])

        
        # print(torch.ones(self.know_length,1).size())
        # print(x.view(1, -1).size())
        # print(x.type())
        # xk = torch.mm(torch.ones(self.know_length, 1), x.view(1, -1))
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = beta.view(-1, 1) * xk
        #print("exam embedding after concat",xk.size(),xk)
        # xk = ko.float().view(-1, 1) * xk
        # print(xk.size())
        # print(alpha.size())
        # xk = torch.mm(alpha, xk).view(-1)
        # thresh, idx = alpha.topk(5)
        # alpha = (alpha >= thresh[0, 4]).float()
        # xk = alpha.view(-1, 1) * xk
        # xk = Variable(torch.zeros_like(x)).expand(self.know_length, -1)
        #print("h_befor return first:",h)
        #print("xk",xk.size(),xk)
        #print("xk.unsqueeze(0)",xk.unsqueeze(0).size(),xk.unsqueeze(0))
        #print("before h",h.size(),h)
        _, h = self.rnn(xk.unsqueeze(0), h)
        #print("after h",h.size(),h)
        #print("h_befor return second:",h)
        return predict_score.view(1), h

class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Networks for Knowledge Tracing at WWW'2017
    """

    @staticmethod
    def add_arguments(parser):
        print("Test")
        #RNN.add_arguments(parser)
        #parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        #parser.add_argument('--knows', default='data/know_list.txt', help='numbers of knowledge concepts')
        #parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        #parser.add_argument('-l', '--num_layers', type=int, default=2, help='#topic rnn layers')
        # parser.add_argument('-es', '--erase_vector_size', type=float, default=25, help='erase vector emb size')
        # parser.add_argument('-as', '--add_vector_size', type=float, default=25, help='add vector emb size')

    def __init__(self, args):
        super(DKVMN, self).__init__()
        self.args = args
        know_dic = open(args.knows).read().split('\n')
        args.kcnt = len(know_dic)
        self.kcnt = args.kcnt
        self.valve_size = args.knowledge_hidden_size * 2

        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt, args.knowledge_hidden_size)
        # student seq module
        self.seq_model = DKVMNSeqModel(args.knowledge_hidden_size, 30, args.kcnt, args.seq_hidden_size,
                                       self.valve_size)

    def forward(self, o_e, score, hidden=None):
        # print(knowledge)
        otEmbeddingResize = nn.Linear(len(o_e), self.kcnt)
        knowledge=otEmbeddingResize(o_e)
        
        expand_vec = knowledge.float().view(-1) * score
        # print(expand_vec)
        cks = torch.cat([knowledge.float().view(-1), expand_vec]).view(1, -1)
        # print(cks)

        knowledge = self.knowledge_model(knowledge)

        s, h = self.seq_model(cks, knowledge, score, hidden)
        return s, h


class DKVMNSeqModel(nn.Module):
    """
    DKVMN seq model
    """

    def __init__(self, know_emb_size, know_length, kcnt, seq_hidden_size, value_size):
        super(DKVMNSeqModel, self).__init__()
        self.know_emb_size = know_emb_size
        self.know_length = know_length
        self.seq_hidden_size = seq_hidden_size
        # self.erase_size = erase_size
        # self.add_size = add_size
        self.value_size = value_size

        # knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # read process embedding module
        self.ft_embedding = nn.Linear(self.seq_hidden_size + self.know_emb_size, 50)
        self.score_layer = nn.Linear(50, 1)

        # write process embedding module
        # erase_size = add_size = seq_hidden_size
        self.cks_embedding = nn.Linear(kcnt * 2, self.value_size)
        self.erase_embedding = nn.Linear(self.value_size, self.seq_hidden_size)
        self.add_embedding = nn.Linear(self.value_size, self.seq_hidden_size)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

    def forward(self, cks, kn, s, h):
        if h is None:
            h = self.h_initial.view(self.know_length * self.seq_hidden_size)

        # calculate alpha weights of knowledges using dot product
        alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
        alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

        # read process
        rt = torch.mm(alpha, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        com_r_k = torch.cat([rt, kn.view(-1)]).view(1, -1)
        # print(com_r_k.size())
        ft = torch.tanh(self.ft_embedding(com_r_k))
        predict_score = torch.sigmoid(self.score_layer(ft))

        # write process
        vt = self.cks_embedding(cks)
        et = torch.sigmoid(self.erase_embedding(vt))
        at = torch.tanh(self.add_embedding(vt))
        ht = h * (1 - (alpha.view(-1, 1) * et).view(-1))
        h = ht + (alpha.view(-1, 1) * at).view(-1)
        return predict_score.view(1), h



#######
# knowledge Representation module
#######
class KnowledgeModel(nn.Module):
    """
    Transform Knowledge index to knowledge embedding
    """

    def __init__(self, know_len, know_emb_size):
        super(KnowledgeModel, self).__init__()
        #print("test",know_len, know_emb_size)
        self.knowledge_embedding = nn.Linear(know_len, know_emb_size)

    def forward(self, knowledge):
        return self.knowledge_embedding(knowledge.float().view(1, -1))



class LSTMM(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--emb_file', default='data/emb_50.txt',
                            help='pretrained word embedding')
        parser.add_argument('--seq_hidden_size', '-hs', type=int, default=50,
                            help='sequence embedding size')
        parser.add_argument('--topic_size', '-ts', type=int, default=50,
                            help='topic embedding size')
        parser.add_argument('--score_mode', '-s',
                            choices=['concat', 'double'], default='double',
                            help='way to combine topics and scores')

    def __init__(self, args):
        super(LSTMM, self).__init__()
        self.args = args
        #wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        #self.words = words
        #self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        #self.embedding.weight.data.copy_(torch.from_numpy(embs))
        self.seq_model = SeqModel(args.topic_size, args.seq_hidden_size,
                                  args.score_mode)

    def forward(self, o_e, score, time, hidden=None):
        otEmbeddingResize = nn.Linear(len(o_e), 100)
        topic_v = otEmbeddingResize(o_e)
        
        #self.embedding(topic).mean(0, keepdim=True)
        s, hidden = self.seq_model(topic_v, score, hidden)
        return s, hidden



class SeqModel(nn.Module):
    """
    做题记录序列的RNN（GRU）单元
    """

    def __init__(self, topic_size, seq_hidden_size, score_mode, num_layers=1):
        super(SeqModel, self).__init__()
        self.topic_size = seq_hidden_size
        self.seq_hidden_size = topic_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(topic_size * 2 + 1, seq_hidden_size, num_layers)
        self.score = nn.Linear(seq_hidden_size + topic_size, 1)

    def forward(self, v, s, h):
        if h is None:
            h = self.default_hidden()
        pred_v = torch.cat([v, h.view(-1)])
        score = self.score(pred_v.view(1, -1))
        
        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v),
                           v * (s < 0.5).type_as(v).expand_as(v)])
        #print(self.score_mode,x, s)
        x = torch.cat([x, s])
        
        _, h = self.rnn(x.view(1, 1, -1), h)
        return score.view(1), h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))



class EKTM(nn.Module):
    """
    Knowledge Tracing Model with Markov property combined with exercise texts and knowledge concepts
    """
    @staticmethod
    def add_arguments(parser):
        print("Test")
        #RNN.add_arguments(parser)
        #parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        #parser.add_argument('-kc', '--kcnt', type=int, default=0, help='numbers of knowledge concepts')
        #parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        #parser.add_argument('-l', '--num_layers', type=int, default=1, help='#topic rnn layers')
        # parser.add_argument('-hs', '--seq_hidden_size', type=int, default=50, help='student seq emb size')
        # parser.add_argument('-ts', '--topic_size', type=int, default=50, help='exercise emb size')
        # parser.add_argument('-s', '--score_mode', choices=['concat', 'double'], default='double',
        #                     help='way to combine exercise and score')

    def __init__(self, args):
        super(EKTM, self).__init__()
        self.args = args
        know_dic = open(args.knows).read().split('\n')
        args.kcnt = len(know_dic)
        self.kcnt = args.kcnt
        """
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        if args['kcnt'] == 0:
            know_dic = open('data/firstknow_list.txt').read().split('\n')
            args['kcnt'] = len(know_dic)
        self.kcnt = args['kcnt']
        """
        # knowledge embedding module
        #convert knowledge to knowledge embedding size. Linear transformation
        self.knowledge_model = KnowledgeModel(self.kcnt, args.knowledge_hidden_size)

        # exercise embedding module
        wcnt = 0
        emb_size = 404
        self.topic_model = TopicRNNModel(wcnt, emb_size, args.topic_size, num_layers=args.num_layers)

        #self.topic_model.load_emb(embs)

        # student seq module
        self.seq_model = EKTSeqModel(args.topic_size, args.knowledge_hidden_size, args.kcnt,
                                     args.seq_hidden_size, args.score_mode)

    def forward(self, co_e, ex_e, score, time, hidden=None):
        # print(knowledge.size())
        #otEmbeddingResize = nn.Linear(len(co_e), self.kcnt).to('cuda:0')
        #co_e=otEmbeddingResize(co_e.float().view(1, -1))
        co_e = self.knowledge_model(co_e)
        
        # print(knowledge.size())
        #exEmbeddingResize = nn.Linear(len(ex_e), 100).to('cuda:0')
        #ex_e = exEmbeddingResize(ex_e)


        topic_h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(ex_e.view(1,1,-1), topic_h)
        s, h = self.seq_model(topic_v[0], co_e, "", score, hidden)
        return s, h
        
class EKTSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, score_mode, num_layers=1):
        super(EKTSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.score_mode = score_mode
        self.num_layers = num_layers
        # self.with_last = with_last

        # Knowledge memory matrix
        
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)
        #print(self.knowledge_memory)
        # Student seq rnn
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(self.topic_size * 2 + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)

    def forward(self, ex_e, co_e, ko, s, h, beta=None):
        #kn = outcome onehot Encoding
        #v = exam embedding
        #v: topic vector
        #kn: knowledge embedding
        #ko: actual knowledge
        if h is None:
             h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
             length = Variable(torch.FloatTensor([0.]))

        # calculate alpha weights of knowledges using dot product
        # print(self.knowledge_memory.size())
        # print(kn.view(-1, 1))
        #print("beta", beta)
        #print("knowledge_memory ", self.knowledge_memory.size())
        #print("co_e.view ",co_e.view(-1, 1).size())
        
        if beta is None:
            alpha = torch.mm(self.knowledge_memory, co_e.view(-1, 1)).view(-1)
            #print("alpha", alpha.size())
            beta = nn.functional.softmax(alpha.view(1, -1), dim=-1)
            #print("beta", beta.size())
            # print(beta.argmax(1))
        
        # print(alpha.size())

        #print("h.view ", h.view(self.know_length, self.seq_hidden_size).size())
        # print(h.type())
        # predict score at time t
        hkp = torch.mm(beta, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        #print("hkp ", hkp.size())
        pred_v = torch.cat([hkp, ex_e]).view(1, -1)
        #print("pred_v ", pred_v.size())
        predict_score = self.score_layer(pred_v)
        #print(predict_score)
        #exit()
        # seq states update
        if self.score_mode == 'concat':
            x = ex_e
        else:
            #x = torch.cat([ex_e * (s >= 0.5).type_as(ex_e).expand_as(ex_e),
            #               ex_e * (s < 0.5).type_as(ex_e).expand_as(ex_e)])
            x = torch.cat([ex_e.view(-1),
                       (ex_e * (s > 0.5).type_as(ex_e).
                        expand_as(ex_e).type_as(ex_e)).view(-1)])
        x = torch.cat([x, s])
        
        # print(x.size())
        # print(torch.ones(self.know_length,1).size())
        # print(x.view(1, -1).size())
        # print(x.type())
        # xk = torch.mm(torch.ones(self.know_length, 1), x.view(1, -1))
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = beta.view(-1, 1) * xk
        # xk = ko.float().view(-1, 1) * xk
        # print(xk.size())
        # print(alpha.size())
        # xk = torch.mm(alpha, xk).view(-1)
        # thresh, idx = alpha.topk(5)
        # alpha = (alpha >= thresh[0, 4]).float()
        # xk = alpha.view(-1, 1) * xk
        # xk = Variable(torch.zeros_like(x)).expand(self.know_length, -1)
        #_, h = self.rnn(x.view(1, 1, -1), h)
        _, h = self.rnn(xk.unsqueeze(0), h)
        return predict_score.view(1), h



class EKTA(nn.Module):
    """
    Knowledge Tracing Model with Attention mechnaism combined with exercise texts and knowledge concepts
    """
    @staticmethod
    def add_arguments(parser):
        print("EKTA")
        #RNN.add_arguments(parser)
        #parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        #parser.add_argument('-kc', '--kcnt', type=int, default=0, help='numbers of knowledge concepts')
        #parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        #parser.add_argument('-l', '--num_layers', type=int, default=1, help='#topic rnn layers')
        #parser.add_argument('-w', '--workspace', default="test", help='#topic rnn layers')
        #parser.add_argument('-c', '--command', default="config", help='#topic rnn layers')

    def __init__(self, args):
        super(EKTA, self).__init__()
        self.args = args
        know_dic = open(args.knows).read().split('\n')
        args.kcnt = len(know_dic)
        self.kcnt = args.kcnt
        """
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        if args['kcnt'] == 0:
            know_dic = open('data/firstknow_list.txt').read().split('\n')
            args['kcnt'] = len(know_dic)
        self.kcnt = args['kcnt']
        """
        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt, args.knowledge_hidden_size)
        emb_size = 50
        # exercise embedding module
        #self.topic_model = TopicRNNModel(0, emb_size, args.topic_size, num_layers=args.num_layers)

        #self.topic_model.load_emb(embs)
        
        # student seq module
        self.seq_model = EKTAttnSeqModel(args.topic_size, args.knowledge_hidden_size, args.kcnt,
                                         args.seq_hidden_size, args.k, args.score_mode)

    def forward(self, co_e, ex_e, score, time, hidden=None, alpha=False):
        # print(knowledge.size())
        knowledge = co_e
        exEmbeddingResize = nn.Linear(len(ex_e), 100)
        topic_v = exEmbeddingResize(ex_e)
        k = self.knowledge_model(knowledge)
        # print(knowledge.size())
        #topic_h = self.topic_model.default_hidden(1)
        #topic_v, _ = self.topic_model(ex_e.view(-1, 1), topic_h)

        s, h, a = self.seq_model(topic_v, k, knowledge, score, hidden)
        if hidden is None:
            hidden = h, topic_v, h
        else:
            _, vs, hs = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            hidden = h, vs, hs

        if alpha:
            return s, hidden, a
        else:
            return s, hidden



class EKTAttnSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, k, score_mode, num_layers=1):
        super(EKTAttnSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.score_mode = score_mode
        self.num_layers = num_layers
        self.k = k
        # self.with_last = with_last

        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # Student seq rnn
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(self.topic_size * 2 + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)
        self.k = k

    def forward(self, v, kn, ko, s, hidden):
        if hidden is None:
            h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
            attn_h = self.h_initial
            length = Variable(torch.FloatTensor([0.]))
            beta = None

        else:

            h, vs, hs = hidden

            # calculate beta weights of seqs using dot product
            beta = torch.mm(vs, v.view(-1, 1)).view(-1)
            beta, idx = beta.topk(min(len(beta), self.k), sorted=False)
            beta = nn.functional.softmax(beta.view(1, -1), dim=-1)
            length = Variable(torch.FloatTensor([beta.size()[1]]))

            hs = hs.view(-1, self.know_length * self.seq_hidden_size)
            attn_h = torch.mm(beta, torch.index_select(hs, 0, idx)).view(-1)

        # calculate alpha weights of knowledges using dot product
        alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
        alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

        hkp = torch.mm(alpha, attn_h.view(self.know_length, self.seq_hidden_size)).view(-1)
        pred_v = torch.cat([v, hkp]).view(1, -1)
        predict_score = self.score_layer(pred_v)

        # seq states update
        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v), v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        # print(x.size())
        # print(torch.ones(self.know_length,1).size())
        # print(x.view(1, -1).size())
        # print(x.type())
        # xk = torch.mm(torch.ones(self.know_length, 1), x.view(1, -1))
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = alpha.view(-1, 1) * xk
        # xk = ko.float().view(-1, 1) * xk
        # xk = torch.mm(alpha, xk).view(-1)

        _, h = self.rnn(xk.unsqueeze(0), h)
        return predict_score.view(1), h, beta



class TopicRNNModel(nn.Module):
    """
    双向RNN（GRU）建模题面
    """

    def __init__(self, wcnt, emb_size, topic_size, num_layers=2):
        super(TopicRNNModel, self).__init__()
        self.num_layers = num_layers
        #self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        if num_layers > 1:
            self.emb_size = topic_size
            self.rnn = nn.GRU(emb_size, topic_size, 1,
                              bidirectional=True,
                              dropout=0.1)
            self.output = nn.GRU(topic_size * 2,
                                 topic_size, num_layers - 1,
                                 dropout=0.1)
        else:
            self.emb_size = topic_size // 2
            self.rnn = nn.GRU(emb_size, topic_size // 2, 1,
                              bidirectional=True)

    def forward(self, input, hidden):
        #x = self.embedding(input)
        x=input
        #print(x.size())
        #print(hidden[0].size())
        # exit(0)
        y, h1 = self.rnn(x, hidden[0])
        if self.num_layers > 1:
            y, h2 = self.output(y, hidden[1])
            return y[-1], (h1, h2)
        else:
            y, _ = torch.max(y, 0)
            return y, (h1, None)

    def default_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.emb_size)), \
            Variable(torch.zeros(self.num_layers - 1,
                                 batch_size, self.emb_size)) \
            if self.num_layers > 1 else None

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))

if __name__ == '__main__':
    f = open("LSTM.json", 'r')
    model = OKT(json.load(f))
    
    #text = Variable(torch.LongTensor(sample2))
    #print("text.float().view(1, -1)",text.float().view(1, -1))
    #print("text",text)
    score = Variable(torch.FloatTensor([1]))
    #topic_v = self.embedding(topic).mean(0, keepdim=True)
    #print(topic_v)
    #s, h = model(text, score, None)
    
    

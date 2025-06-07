from preprocessing import *
from csv import reader
from OKT import *
#from DKVMN import *
import json
import time
from util import save_snapshot, load_last_snapshot, Variable
import argparse
from argparse import Namespace
import os
from sklearn import metrics
from sklearn.model_selection import KFold
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainn:

    def getExamEmbedding(self,examRow,coEncoding):
        #coText = "In course "+data.courseName+" with outcome "+data.courseOutcome
        bertEncoding = coEncoding #bertModel.encode(coText)
        examFeatures = examRow.examFeatures.split(',')
        examFeatures = np.array(examFeatures)
        bertEncoding = np.array(bertEncoding)
        examFeatures = examFeatures.astype('float32')
        encoding = np.concatenate((examFeatures, bertEncoding))
        
        encoding = torch.FloatTensor(encoding)
        tranformer = nn.Linear(len(encoding), 100)
        encoding = tranformer(encoding)
        
        return encoding

    def getCOEmbedding(self,examRow,coEncoding):
        #coText = "In course "+data.courseName+" with outcome "+data.courseOutcome
        bertEncoding = coEncoding
        mappings = examRow.mapping.split(',')
        mappings.append(int(examRow.threshold)/100)
        mappings.append(int(examRow.target)/100)
        mappings = np.array(mappings)
        bertEncoding = np.array(bertEncoding)
        mappings = mappings.astype('float32')
        encoding = np.concatenate((mappings, bertEncoding))
        #print(len(encoding),encoding)
        
        encoding = torch.FloatTensor(encoding)
        tranformer = nn.Linear(len(encoding), 100)
        encoding = tranformer(encoding)
        
        return encoding
        
    def getCOEmbeddingNoMapping(self,examRow):
        coText = "In course "+examRow.courseName+" with outcome "+examRow.courseOutcome
        bertEncoding = bertModel.encode(coText)
        bertEncoding = np.array(bertEncoding)
        return bertEncoding
        
    def getCOOnehotEncoding(self,examRow):
        if examRow.courseOutcomeID in self.topic_dic:
            knowledge = self.topic_dic[examRow.courseOutcomeID]
        else:
            knowledge = self.zero
        knowledge = Variable(torch.LongTensor(knowledge))
        return knowledge
    
    def run_one_epoch(self,epoch, data,type,fold):
        warnings.filterwarnings("ignore")
        then = time.time()
        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_auc = 0
        user_cnt = 0
        if type=="train":
            userData = data.traindata
        else:
            userData = data.testdata
        for (userID,userValues) in userData:
            #if userID!="AM.EN.U4CSE16221": continue           
            loss = 0
            mae = 0
            acc = 0
            self.optimizer.zero_grad()
            #seq_length = len(userValues)
            h=None
            seq_length=0
            userScoreSummary = {"0":0,"1":0,"2":0,"0acc":0,"1acc":0,"2acc":0}
            predScores = []
            prevCOID = None
            y_train =[0,1]
            y_pred =[0,1]
            for value in userValues:
                examRow = ExamObject(*value)

                seq_length+=1
                prevCOID = examRow.courseOutcomeID

                encoding = data.courseOutcomes[examRow.courseOutcomeID]
                coEmbedding = encoding #self.getCOEmbedding(examRow,encoding)

                examEmbedding = self.getExamEmbedding(examRow,encoding)
                coOneHotEncoding = self.getCOOnehotEncoding(examRow)
                exEmb = torch.FloatTensor(examEmbedding)
                coEmb = torch.FloatTensor(coEmbedding)
                score = torch.FloatTensor([float(int(examRow.attainment))])
                if  args.model== "OKT":
                    s, h = self.model(coEmb,exEmb, score, h)
                s_p = s[0]
                att_p = 0 if s_p<0.5 else 1
                y_train.append(examRow.attainment)
                y_pred.append(s_p.item())
                loss += self.MSE(s_p, score)
                m = self.MAE(s_p, score).item()
                mae += m
                acc += (m < 0.5)
                userScoreSummary[str(examRow.attainment)]=userScoreSummary[str(examRow.attainment)]+1
                if (int(examRow.attainment)==int(att_p)):
                    userScoreSummary[str(examRow.attainment)+"acc"]=userScoreSummary[str(examRow.attainment)+"acc"]+1
            if seq_length==0: continue
            

            user_cnt += 1
            curr_auc = metrics.roc_auc_score(y_train, y_pred)
            total_auc +=curr_auc
            loss /= seq_length
            mae /= seq_length
            acc = float(acc) / seq_length

            total_loss += loss.data.item()
            total_mae += mae
            total_acc += acc

            if type=="train":
                loss.backward()
                self.optimizer.step()
            
            now = time.time()
            duration = (now - then) / 60
            scoreSummary = str(userScoreSummary["0acc"])+"/"+str(userScoreSummary["0"])+", "+str(userScoreSummary["1acc"])+"/"+str(userScoreSummary["1"])
            user_cnt = 1 if user_cnt==0 else user_cnt
            print('%s: ,fold,%d,epoch %d, userid: %s, user:%d/%d, exams: %d (%.2f seqs/min), total loss: %.6f,curr loss %.6f, mae %.6f, total acc %.6f, curr acc %.6f, total auc %.6f, curr auc %.6f, summary %s, predicted scores: %s ' %
                 (type,fold,epoch,userID,user_cnt,self.total_user_cnt,seq_length, (0 if duration==0 else ((user_cnt-1) % self.print_every + 1)/duration),
                  total_loss/user_cnt,loss, total_mae/user_cnt, total_acc/user_cnt,acc,total_auc/user_cnt,curr_auc,scoreSummary,predScores))
            writeTofile('%s:fold,%d,epoch,%d, userid: %s, user: %d/%d, exams, %d (%.2f seqs/min), total loss, %.6f,curr loss, %.6f, mae, %.6f, total acc, %.6f, curr acc, %.6f, total auc, %.6f, curr auc, %.6f, summary, %s, predicted scores: %s ' %
                             (type,fold,epoch,userID,user_cnt,self.total_user_cnt,seq_length, ((user_cnt-1) % self.print_every + 1)/duration,
                              total_loss/user_cnt,loss, total_mae/user_cnt, total_acc/user_cnt,acc,total_auc/user_cnt,curr_auc,scoreSummary,predScores),type+"_log.csv")

            then = now
        return (total_loss/user_cnt,total_acc/user_cnt,total_mae/user_cnt,total_auc/user_cnt)

    def __init__(self, args,data,fold):
        self.topic_dic = {}
        f = open("conf.json", 'r')
        args = Namespace(**json.load(f))
        with open(args.knowids, 'r') as file:
            concepts = [line.strip().split('|')[0] for line in file.readlines()]
        concepts_reshaped = [[concept] for concept in concepts]
        encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = encoder.fit_transform(concepts_reshaped)
        one_hot_encoded_list = one_hot_encoded.tolist()
        self.topic_dic = {concept: one_hot_encoded_list[i] for i, concept in enumerate(concepts)}
        self.zero = [0] * len(concepts)

        if args.model=="OKT":
            self.model = OKT(args)


        self.MSE = torch.nn.MSELoss()
        self.MAE = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        start_epoch = load_last_snapshot(self.model, args.workspace,args.model+"_"+str(fold))
        self.total_user_cnt = len(data.traindata)
        self.print_every = 2
        norm_attainment = {0:0, 1:0.33, 2:0.66}
        for epoch in range(start_epoch, args.epoch):
            #self.model.train(True)
            (avg_loss,avg_acc,avg_mae,avg_auc) = self.run_one_epoch(epoch,data,"train",fold)
            writeTofile("training ,fold,%d, epoch, %d, total loss, %.6f, total accuracy, %.6f, total mae, %.6f, total auc,%.6f " % (fold,epoch,avg_loss,avg_acc,avg_mae,avg_auc),"training_result.csv")
            save_snapshot(self.model, args.workspace, '%d' % (epoch),args.model+"_"+str(fold))
            (avg_loss,avg_acc,avg_mae,avg_auc) = self.run_one_epoch(epoch,data,"test",fold)
            writeTofile("testing ,fold,%d, epoch, %d, total loss, %.6f, total accuracy, %.6f, total mae, %.6f, total auc,%.6f " % (fold,epoch,avg_loss,avg_acc,avg_mae,avg_auc),"test_result.csv")

if __name__ == '__main__':
    bertModel = SentenceTransformer('./fine-tuned-bert')
    parser = argparse.ArgumentParser(description='OKT')
    #parser.add_argument('-w', '--workspace', default="data", help='#workspace folder')
    #parser.add_argument('-m', '--model', default="OKT", help='#name the model')
    args = parser.parse_args()
    f = open("conf.json", 'r')
    args = Namespace(**json.load(f))
    args.workspace = os.path.join(os.getcwd(),args.workspace)
    data = DataLoading(args)
    all_data = [(k, v) for k, v in data.users.items()]
    
    kfold = KFold(n_splits=4, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_data)):
        print(fold,len(train_ids),len(test_ids))
        data.traindata = torch.utils.data.dataset.Subset(all_data,train_ids)
        data.testdata = torch.utils.data.dataset.Subset(all_data,test_ids)
        trainn(args,data,fold)
   

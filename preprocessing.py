import os
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from csv import reader
from sentence_transformers import SentenceTransformer

bertModel = SentenceTransformer('./fine-tuned-bert')

class ExamObject:
    def __init__(self, *args):
        self.userUID = args[0] if len(args) > 0 else None
        self.courseOutcomeID = args[1] if len(args) > 1 else None
        self.courseName = args[2] if len(args) > 5 else None
        self.courseOutcome = args[3] if len(args) > 6 else None
        self.threshold = args[4] if len(args) > 2 else None
        self.target = args[5] if len(args) > 3 else None
        self.mapping = args[6] if len(args) > 4 else None
        self.examID = args[7] if len(args) > 7 else None
        self.examFeatures = args[8] if len(args) > 8 else None
        self.percentage = args[9] if len(args) > 9 else None
        self.attainment = args[10] if len(args) > 10 else None
        
class DataLoading:
    def __init__(self, args):
        self.workspace = args.workspace
        self.users = {}
        self.courseOutcomes={}
        self.data_file = args.data_file
        self.exam_feature_file = args.exam_feature_file
        self.co_emb_file = args.co_emb_file
        self.users = self.readData()

    def generateExamFeatures(self):
        df = pd.read_csv(self.data_file)
        examFeatureFilename = self.exam_feature_file
        coEmbeddingFilename = self.co_emb_file
        df = df.sort_values(by = ['UserUID', 'Semester',"Exam_type"], ascending = [True, True, True], na_position = 'first')
        df.loc[df['DifficultyLevel'].isin([1]),'Difficulty'] = 'Easy'
        df.loc[df['DifficultyLevel'].isin([2]),'Difficulty'] = 'Medium'
        df.loc[df['DifficultyLevel'].isin([3]),'Difficulty'] = 'Hard' 
        print(df['Difficulty'].unique())
        data=df
        data['attainment']=data['percentage'].apply(lambda x: 0 if x<0.5 else 1)
        data[['Exam_type','Semester']] = data[['Exam_type','Semester']].astype(int)
        data['mappinglist'] = data.Mapping.apply(lambda x: [int(y) for y in x[0:].split(',')])
        one_hot_encoded_data = pd.get_dummies(data, columns = ['programCode', 'AcademicYear','Exam_type','Difficulty','Semester'])
        #one_hot_encoded_data.info()
        exam_data = one_hot_encoded_data[["UserUID","CourseOutcomeID","CourseName","CourseOutcome","examID","percentage","attainment","Threshold","Target","Mapping"
        ,"Semester_1","Semester_2","Semester_3","Semester_4","Semester_5","Semester_6","Semester_7","Semester_8"
        ,"Exam_type_1","Exam_type_2","Exam_type_3","Exam_type_4"
        ,"Difficulty_Easy","Difficulty_Hard","Difficulty_Medium"
        ,"AcademicYear_2017-2018","AcademicYear_2018-2019"
        ]]
        exam_data["examFeatures"] = exam_data[exam_data.columns[10:]].values.tolist()
        exam_data["examFeatures"] = exam_data["examFeatures"].apply(lambda x: [int(i) for i in x])
        exam_data["examFeatures"] = exam_data["examFeatures"].apply(lambda x: ','.join(map(str, x)))
        #exam_data['examFeatures'] = exam_data['examFeatures'].astype(int)
        exam_data[["UserUID","CourseOutcomeID","CourseName","CourseOutcome","Threshold","Target"
                   ,"Mapping","examID","examFeatures","percentage","attainment"]].to_csv(examFeatureFilename)
        
        unique_df = exam_data.drop_duplicates(subset=['CourseOutcomeID', 'CourseName','CourseOutcome',"Threshold","Target","Mapping"])
        unique_df["concat_co"] = "In course "+unique_df.CourseName+" with outcome "+unique_df.CourseOutcome 
        unique_df['co_embedding'] = unique_df['concat_co'].apply(lambda x: torch.FloatTensor(bertModel.encode(x)))
        unique_df['co_embedding'] = unique_df['co_embedding'].apply(lambda x: x.tolist())
        unique_df['mappinglist'] = unique_df.Mapping.apply(lambda x: [int(y) for y in x[0:].split(',')])
        unique_df["TTList"] = unique_df[["Threshold","Target"]].values.tolist()
        unique_df['ConcatEmbedding'] = unique_df['mappinglist'] + unique_df['TTList']+unique_df['co_embedding']
        
        #Apply dimensionality reduction using PCA
        X = np.vstack(unique_df['ConcatEmbedding'].values)  # Convert list of arrays to 2D array
        # Initialize PCA to reduce dimensionality to 100
        pca = PCA(n_components=100)
        # Fit PCA model to the data
        pca.fit(X)
        # Transform the data to the new lower-dimensional space
        X_reduced = pca.transform(X)
        reduced_arrays = [row for row in X_reduced]
        unique_df['co_embedding'] = reduced_arrays
        unique_df['co_embedding'] = unique_df['co_embedding'].apply(lambda x: x.tolist())
        
        
        unique_df[["CourseOutcomeID","CourseOutcome","co_embedding"]].to_csv(coEmbeddingFilename)
        return examFeatureFilename, coEmbeddingFilename
           
    def readData(self):
        users = {}
        examFeatureFilename, coEmbeddingFilename = self.generateExamFeatures()
        self.courseOutcomes = self.readCourseOutcomeEmbedding(coEmbeddingFilename)
        with open(examFeatureFilename, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            headings = next(csv_reader)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                #print(row[2])
                # row variable is a list that represents a row in csv
                userID = row[1]
                if userID in users:
                    users[userID].append(row[1:])
                else:
                    users[userID]=[row[1:]]
        
        return users
	
    def readCourseOutcomeEmbedding(self,filename):
        courseOutcomes = {}
        with open(filename, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            headings = next(csv_reader)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                #print(row[2])
                courseOutcomeID = row[1]
                embStr = row[3].replace("[","").replace("]","").replace(" ","")
                courseOutcomes[courseOutcomeID]=[float(i.strip()) for i in embStr.split(',')]
                #print(len(courseOutcomes[courseOutcomeID]))
        
        
        new_file=open(os.path.join(self.workspace,"emb_co.txt"),mode="w",encoding="utf-8")
        new_file.write(str(len(courseOutcomes))+" "+"50"+"\n")
        for coID,emb in courseOutcomes.items():
            x_arrstr = np.char.mod('%f', emb)
            x_str = " ".join(x_arrstr)
            new_file.write(coID+" "+x_str+"\n")
        new_file.close()
        return courseOutcomes

    def readExamEmbedding(self):
        exams = {}
        with open(os.path.join(self.workspace,'Exam_Embedding.csv'), 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            headings = next(csv_reader)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                examID = row[1]
                embStr = row[2].replace("[","").replace("]","").replace(" ","")
                exams[examID]=[float(i.strip()) for i in embStr.split(',')]
        return exams

def writeTofile(text,filename):
    file1 = open(filename, "a")  # append mode # "result.txt"
    file1.write(text)
    file1.write("\n")
    file1.close()

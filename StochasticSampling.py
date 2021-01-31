import random
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

#Algorithm to select points using inverse transform sampling and SVM(binary classification)
class StochasticSelection():

	#Aplly SVM
    def runSVM(self,dfTest,dfTrain):

        #Break the training set into two parts:
        #X = attributes
        #y = classes
        X = dfTrain.drop(["image_name","Class"],axis=1)
        y = dfTrain["Class"]

        #fit the classifier 
        #The parameters gamma and C may be different for other cases, you can try values from ([0.01;100]), in my case these reached the best accuracy. 
        clf = svm.SVC(kernel="rbf", gamma=0.05,C=1)
        clf.fit(X,y)

        # decision_function will return for each dataframe element its distance to the SVM hyperplane      
        predictList = clf.decision_function(test)
        
        #If Class > 0, belongs to class 1
        #If Class < 0, belongs to class -1
        dfTest["Class"] = predictList
        dfTest = dfTest.sort_values(by=['Class'],ascending=False )
        
        # Return the test set associated with the predicted classes
        return dfTest



    def selectPtsSVMInverseTransf(self,dfClassified,samples):

    	#Calculate the probability of sampling a point that belongs to class 1
        positiveRowIndex = dfClassified.loc[dfClassified['Class'] > 0.0].index
        dfPositive = dfClassified.loc[positiveRowIndex, :]
        dfPositive = self.calcProb(dfPositive)
       
        #Calculate the probability of sampling a point that belongs to class -1
        negativeRowIndex = dfClassified.loc[dfClassified['Class'] < 0.0].index
        dfNegative = dfClassified.loc[negativeRowIndex, :]
        dfNegative = self.calcProb(dfNegative)
    
    	#Sample X(qtdPos) points from class 1
        self.df1 = self.selectPtsProb(dfPositive,qtdPos,"Positive",randomMin=0)
        
		#Sample Y(qtdNeg) points from class -1
        self.df2 = self.selectPtsProb(dfNegative,qtdNeg,"Negative",randomMin=0)
        
        dfPoints = pd.concat([self.df1, self.df2])
        
        return dfPoints

    def calcProb(self,df):

    	df['Class'] = df['Class'].abs()
        sumSVMDistance = df["Class"].values.sum()
        df = df.sort_values(by=['Class'], ascending=True)
		
		listPj = []
        listCumulativeSumPj= []
        listDistance = df["Class"].tolist()
        last = 0
       
        for distance in listDistance:

        	#Probability to select a point Pj
            pj = distance/sumSVMDistance
            listPj.append(pj)

            sumPj = pj + last
            listCumulativeSumPj.append(sumPj)
            last = sumPj

        df['Pj'] = listPj
        #Saves the cumulative sum of the probability until each point P
        df['CumulativeSumPj'] = listCumulativeSumPj

        return df


    def selectPtsProb(self,df,samples,controle,flag,randomMin):
        newRows = pd.DataFrame(columns=df.columns)

        listCumulativeSum = df["CumulativeSumPj"].tolist()
        
        for x in range(0,samples):
        	#Generates a random number between [0,1]
            u = random.uniform(randomMin, 1)
            for cumulativeSum in listCumulativeSum:

            	#Selects the point P if the cumulative sum of probability until P is greater than the generated random number 
                if u < cumulativeSum:
                    dfTemp = df.loc[df['CumulativeSumPj']== cumulativeSum]
                    newRows = pd.concat([newRows,dfTemp])
                    listCumulativeSum.remove(cumulativeSum)
                    df = df.drop(df.loc[df['CumulativeSumPj']==cumulativeSum].index,axis =0)
                    break
                    
        return newRows
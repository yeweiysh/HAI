import pandas as pd
import numpy as np
import fileinput
import json
from scipy.stats import beta
from scipy import stats
import matplotlib.pyplot as plt
import re
import networkx as nx
import math
import random
import pickle
import AnalysisFunctions as plf
from scipy.stats import wilcoxon
from statistics import mean
from scipy.stats import pearsonr
from cpt_valuation import evaluateProspectVals
from sklearn.metrics import precision_recall_fscore_support
import sys
from numpy import linalg as LA
from networkx.algorithms.dag import is_aperiodic
from networkx.algorithms.components import is_strongly_connected
# from networkx.algorithms.centrality import eigenvector_centrality
from itertools import groupby
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm

class HumanDecisionModels:
    def __init__(self,teamId,directory):

        #Constants
        self.numQuestions = 45
        self.trainingSetSize = 15
        self.testSetSize = 30
        self.numAgents = 4

        self.numCentralityReports = 9

        self.c = 4
        self.e = -1
        self.z = -1

#         Other Parameters
        self.influenceMatrixIndex = 0
        self.machineUseCount = [-1, -1, -1, -1]
        self.firstMachineUsage = [-1, -1, -1, -1]

        # Preloading of the data
        eventLog = pd.read_csv(directory+"event_log.csv", sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])

        teamSubjects = pd.read_csv(directory+"team_has_subject.csv",sep=',',quotechar="|",names=["id","teamId","sender_subject_id"]).drop('id',1)

        elNoMessage =  eventLog[(eventLog['event_type'] == "TASK_ATTRIBUTE")]

        elNoMessage["sender_subject_id"] = pd.to_numeric(elNoMessage["sender_subject_id"])

        eventLogWithTeam = pd.merge(elNoMessage, teamSubjects, on='sender_subject_id', how='left')
        eventLogTaskAttribute = eventLogWithTeam[(eventLogWithTeam['event_type'] == "TASK_ATTRIBUTE") & (eventLogWithTeam['teamId'] == teamId)]
        #Extract data from event_content column
        newEventContent = pd.DataFrame(index=np.arange(0, len(eventLogTaskAttribute)), columns=("id","stringValue", "questionNumber","questionScore","attributeName"))
        self.questionNumbers = list()

        for i in range(len(eventLogTaskAttribute)):
            newEventContent.id[i] = eventLogTaskAttribute.iloc[i]["id"]
            newEventContent.stringValue[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[0].split(":")[1].replace('"', '')
            newEventContent.questionNumber[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[1].split(":")[1]
            if newEventContent.questionNumber[i] not in self.questionNumbers:
                self.questionNumbers.append(newEventContent.questionNumber[i])
            newEventContent.questionScore[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[3].split(":")[1]
            newEventContent.attributeName[i] =eventLogTaskAttribute.iloc[i]["event_content"].split("||")[2].split(":")[1]

        self.questionNumbers = self.questionNumbers[1 :]
        # print(self.questionNumbers)
        eventLogWithAllData = pd.merge(eventLogTaskAttribute,newEventContent,on='id', how ='left')

        # print(eventLogWithAllData)

        # Load correct answers
        with open(directory+"jeopardy45.json") as json_data:
                d = json.load(json_data)
        self.correctAnswers = list()
        self.options = list()

        for i in range(0, self.numQuestions):
            self.correctAnswers.append(d[int(float(self.questionNumbers[i]))-1]['Answer'])
            self.options.append(d[int(float(self.questionNumbers[i]))-1]['value'])

        # print(self.options)


        allIndividualResponses = eventLogWithAllData[eventLogWithAllData['extra_data'] == "IndividualResponse"]

        self.lastIndividualResponsesbyQNo = allIndividualResponses.groupby(['sender', 'questionNumber'], as_index=False, sort=False).last()
        # print(self.lastIndividualResponsesbyQNo["event_content"])
        # print(len(self.lastIndividualResponsesbyQNo))
        # print(self.lastIndividualResponsesbyQNo["questionNumber"])

        self.machineAsked = eventLogWithAllData[eventLogWithAllData['extra_data'] == "AskedMachine"]
        self.machineAskedQuestions = list()
        self.exploreQuestions = list()
        for i in range(len(self.machineAsked)):
            qid = self.machineAsked.iloc[i]['questionNumber']
            # print(qid)
            event1 = self.lastIndividualResponsesbyQNo[self.lastIndividualResponsesbyQNo["questionNumber"] == qid]
            # print(event1["event_content"])
            answ = list()
            for j in range(4):
                answ.append(event1.iloc[j]["event_content"].split("||")[0].split(":")[1].replace('"', ''))
            
            # cnt = 0 
            # correct_ans = d[int(float(qid))-1]['Answer']
            # for ind_ans in answ:
            #     if ind_ans == correct_ans:
            #         cnt += 1
            self.machineAskedQuestions.append(int(float(self.machineAsked.iloc[i]['questionNumber'])))
            if len(set(answ)) != 1:
            # if cnt == 0:
                self.exploreQuestions.append(int(float(qid)))

        # print(self.machineAskedQuestions)
        # print(self.exploreQuestions)


        # Compute the group answer of the team per question
        submissions = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "IndividualResponse") | (eventLogWithAllData['extra_data'] == "GroupRadioResponse") ]
        individualAnswersPerQuestion = submissions.groupby(["questionNumber","sender_subject_id"], as_index=False, sort=False).tail(1)

        # groupR = eventLogWithAllData[eventLogWithAllData['extra_data'] == "GroupRadioResponse"]
        # lastGroupR = groupR.groupby(['sender', 'questionNumber'], as_index=False, sort=False).last()
        # print(groupR)
        self.groupSubmission = pd.DataFrame(index=np.arange(0, len(self.questionNumbers)), columns=("questionNumber","groupAnswer"))
        # print(self.groupSubmission)
        for i in range(0, self.numQuestions):
            ans = ""
            consensusReached = True
            for j in range(0,len(individualAnswersPerQuestion)):
                if (individualAnswersPerQuestion.iloc[j].loc["questionNumber"] == self.questionNumbers[i]):
                    if not ans:
                        ans = individualAnswersPerQuestion.iloc[j].loc["stringValue"]

                    elif (ans != individualAnswersPerQuestion.iloc[j].loc["stringValue"]):
                        consensusReached = False
                        break

            self.groupSubmission.questionNumber[i] = self.questionNumbers[i]
            if (consensusReached):
                self.groupSubmission.groupAnswer[i] = ans
            else:
                self.groupSubmission.groupAnswer[i] = "Consensus Not Reached"

        # Define teammember order
        subjects = pd.read_csv(directory+"subject.csv", sep=',',quotechar="|", names=["sender_subject_id","externalId","displayName","sessionId","previousSessionSubject"])
        teamWithSujectDetails = pd.merge(teamSubjects, subjects, on='sender_subject_id', how='left')
        self.teamMember = teamWithSujectDetails[(teamWithSujectDetails['teamId'] == teamId)]['displayName']
        self.teamSize = len(self.teamMember)
        self.teamArray = list()

        for i in range(self.teamSize):
            self.teamArray.append(self.teamMember.iloc[i])
        
        #         Pre-experiment Survey
        preExperimentData = eventLogWithAllData[eventLogWithAllData['extra_data'] == "RadioField"]
        self.preExperimentRating = list()
        for i in range(0,self.teamSize):
            self.preExperimentRating.append(0)
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]) >0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")]['stringValue'].iloc[0][0:1]))
            self.preExperimentRating[-1]/=15


        # Extracting Machine Usage Information
        self.machineUsed = np.array([False, False, False, False] * self.numQuestions).reshape((self.numQuestions, 4))
        for i in range(self.numQuestions):
            if int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                indxM = self.machineAskedQuestions.index(int(float(self.questionNumbers[i])))
                # print("indxM")
                # print(indxM)
                k = self.teamArray.index(self.machineAsked['sender'].iloc[indxM])
                # print(self.machineAsked['sender'].iloc[indxM])
                # print("k")
                # print(k)
                self.machineUsed[i][int(k)] = True
        


        # print("machineUsed: ")
        # print(self.machineUsed)



        self.teamScore = list()
        self.computeTeamScore()

#         Extract Influence Matrices
        self.agentRatings = list()
        self.memberInfluences = list()
        mInfluences = [0 for i in range(self.teamSize)]
        aRatings = [0 for i in range(self.teamSize)]
        count = 0
        influenceMatrices = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "InfluenceMatrix")]
        influenceMatrixWithoutUndefined = influenceMatrices[~influenceMatrices['stringValue'].str.contains("undefined")]
        finalInfluences = influenceMatrixWithoutUndefined.groupby(['questionScore', 'sender'], as_index=False, sort=False).last()

        for i in range(len(finalInfluences)):
            count +=1
            aR = list()
            mI = list()
            idx = self.teamArray.index(finalInfluences.iloc[i]['sender'])
            for j in range(0, self.teamSize):
                temp = finalInfluences.iloc[i]['stringValue']
#                 Fill missing values
                xy = re.findall(r'Ratings(.*?) Member', temp)[0].split("+")[j].split("=")[1]
                if(xy==''):
                    xy = '0.5'
                yz= temp.replace('"', '')[temp.index("Influences ")+10:].split("+")[j].split("=")[1]
                if(yz == ''):
                    yz = '25'
                aR.append(float(xy))
                mI.append(int(round(float(yz))))
            aRatings[idx]=aR
            mInfluences[idx]=mI
            if(count%self.teamSize == 0):
                self.memberInfluences.append(mInfluences)
                mInfluences = [0 for i in range(self.teamSize)]
                self.agentRatings.append(aRatings)
                aRatings = [0 for i in range(self.teamSize)]

        # Hyperparameters for expected performance (Humans and Agents) - TODO
        self.alphas = [1,1,1,1,1,1,1,1]
        self.betas = np.ones(8, dtype = int)

        #vector c
        self.centralities = [[] for _ in range(self.numQuestions)]

        self.actionTaken = list()
        self.actionTaken1 = list()
        self.computeActionTaken()

    def computeTeamScore(self):
        self.teamScore.append(0)
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i]!=self.correctAnswers[i]:
                self.teamScore[i]+=self.z
            else:
                self.teamScore[i]+=self.c
            if len(np.where(self.machineUsed[i] == True)[0])!=0:
                self.teamScore[i]+=self.e
            self.teamScore.append(self.teamScore[i])
        self.teamScore = self.teamScore[:-1]

    def updateAlphaBeta(self, i, valueSubmitted, correctAnswer):

        if (valueSubmitted == correctAnswer):
            self.alphas[i]+=1
        else:
            self.betas[i]+=1


    def naiveProbability(self, questionNumber, idx):
        expectedPerformance = list()
        individualResponse = list()
        probabilities = list()
        human_accuracy = list()

        machine_accuracy = [None for _ in range(self.numAgents)]
        group_accuracy = 0

        #Save human expected performance based

        for i in range(0,self.teamSize):
            expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
            individualResponse.append(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())
            self.updateAlphaBeta(i,self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any(),self.correctAnswers[idx])
            
            # print("individual answer")
            # print(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())

            ans = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            if ans == self.correctAnswers[idx]:
                human_accuracy.append(1)
            else:
                human_accuracy.append(0)

        if (self.groupSubmission["groupAnswer"].iloc[idx] == self.correctAnswers[idx]):
            group_accuracy = 1

        indxQ = -1
        anyMachineAsked = False
        # print(int(float(questionNumber)))
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            # print("k")
            # print(k)
            anyMachineAsked = True

        # Add expected Performance for Agents
        for i in range(self.teamSize, self.teamSize+self.numAgents):
            expectedPerformance.append(0.5 + 0.5 * beta.mean(self.alphas[i],self.betas[i]))
            # expectedPerformance.append(0.5 + 0.3 * beta.mean(self.alphas[i],self.betas[i]))
            # update alpha beta for that machine

        #Update machine accuracy
        machineAnswer = -1
        if(anyMachineAsked):
            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
            # print(machineAnswer)
            self.updateAlphaBeta(self.getAgentForHuman(k), machineAnswer, self.correctAnswers[idx])
            self.machineUseCount[k]+=1

            machineAnswer = self.options[idx].index(machineAnswer)

            if self.firstMachineUsage[k] == -1:
                self.firstMachineUsage[k] = idx

            if (machineAnswer == self.correctAnswers[idx]):
                machine_accuracy[k] = 1
            else:
                machine_accuracy[k] = 0

            # machine_accuracy[k] = 1

   
        # Conditional Probability
        # Do a bayes update
        denominator = 0
        numerator = [1. for _ in range(len(self.options[idx]))]
        prob_class = 0.25
        prob_rsep = 0
        prob_class_responses = [None for _ in range(len(self.options[idx]))]
        prob_resp_given_class = [None for _ in range(len(self.options[idx]))]

        for opt_num in range(0,len(self.options[idx])):
            prob_resp = 0
            numerator = prob_class
            for person_num in range(0,self.teamSize):
                if individualResponse[person_num] == self.options[idx][opt_num]:
                    numerator *= expectedPerformance[person_num]
                else:
                    numerator *= (1 - expectedPerformance[person_num])/3
                prob_resp += numerator
            prob_resp_given_class[opt_num] = numerator
        prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]

        # print("expectedPerformance: ")
        # print(expectedPerformance)
        #ANSIs this updating agent probabilities?
        # for i in range(self.teamSize):
        #     probabilities.append(expectedPerformance[self.teamSize+i])

        #origin
        l1 = 0
        for i in range(self.teamSize):
            l1 += expectedPerformance[self.teamSize+i]

        for i in range(self.teamSize):
            probabilities.append(1.0 * expectedPerformance[self.teamSize+i]/l1)

        #8 probability values returned
        # first set is for options (sums to 1)
        assert(sum(prob_class_responses) > 0.999 and sum(prob_class_responses) < 1.001)

        #second set is for machines
        # print(prob_class_responses)
        four_agent_prob = [1.0 * expectedPerformance[self.getAgentForHuman(k)] / l1 for k in range(self.teamSize)]
        ap = 1
        for agt_prob in four_agent_prob:
            ap = ap * (1-agt_prob)

        ap = 1 - ap
        prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)] / l1 for k in range(self.teamSize)]
        prob_five_class_responses = prob_class_responses + [ap]
        # print(prob_all_class_responses)
        # prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)] for k in range(self.teamSize)]
        

        return expectedPerformance, prob_all_class_responses,prob_five_class_responses,human_accuracy,group_accuracy,machine_accuracy, machineAnswer


    def updateCentrality(self, influenceMatrixIndex):
        #Compute Eigen Vector Centrality for Humans
        influM = np.zeros((self.teamSize,self.teamSize))
        for i in range(0,self.teamSize):
            ss = 0
            for j in range(0,self.teamSize):
                influM[i][j] = self.memberInfluences[influenceMatrixIndex][i][j]/100
                #influM[i][j] += sys.float_info.epsilon
                ss += influM[i][j]

            for j in range(0,self.teamSize):
                influM[i][j] /= ss


        agentR = np.zeros((self.teamSize,self.teamSize))
        for i in range(0,self.teamSize):
            ss = 0
            for j in range(0,self.teamSize):
                agentR[i][j] = self.agentRatings[influenceMatrixIndex][i][j]
                ss += agentR[i][j]
            if ss != 0:
                for j in range(0,self.teamSize):
                    agentR[i][j] /= ss


        graph = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph.add_edge(i,j,weight=influM[i][j])


        graph1 = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph1.add_edge(i,j,weight=agentR[i][j])

        aperiodic = is_aperiodic(graph1)
        connected = is_strongly_connected(graph1)

        W1 = nx.adjacency_matrix(graph1)
        agent_centralities = nx.eigenvector_centrality(graph1, max_iter=1000, weight="weight")
        eig_centrality1 = list(agent_centralities.values())
        summ = 0
        for eigv in eig_centrality1:
            summ += eigv
        for i in range(4):
            eig_centrality1[i] /= summ


        W = nx.adjacency_matrix(graph)
        # print(W.todense())
        # eigenValues, eigenVectors = np.linalg.eigh(W.todense())
        # print(eigenValues)

        human_centralities = nx.eigenvector_centrality(graph, weight="weight")

        # aa=eigenvector_centrality(graph, weight="weight")
        eig_centrality = list(human_centralities.values())
        summ = 0
        for eigv in eig_centrality:
            summ += eigv
        for i in range(4):
            eig_centrality[i] /= summ


        # print(np.argsort(eig_centrality))
        # print(list(aa.values()))
        # print(eig_centrality)
        #largest = max(nx.strongly_connected_components(graph), key=len)

        #Compute expected performance for machines

        """
        for i in range(0,self.teamSize):
            numerator = 0
            denominator = 0
            for j in range(0,self.teamSize):
                numerator+= self.centralities[j] * self.agentRatings[influenceMatrixIndex][j][i]
                denominator+= self.centralities[j]
            self.centralities.update({self.teamSize+i:numerator/denominator})

        """
        #Check that we have the correct total influence
        for i in range(self.teamSize):
            assert(sum(self.memberInfluences[influenceMatrixIndex][i][j] for j in range(self.numAgents)) == 100)

        #Make a probability

        agent_weighted_centrality_perf = [None for _ in range(self.numAgents)]
        '''
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([self.memberInfluences[influenceMatrixIndex][i][j]/100. for j in range(self.numAgents)])
        '''
        centralities_as_list = [value for value in human_centralities.values()]
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([centralities_as_list[j]*self.agentRatings[influenceMatrixIndex][j][i] for j in range(self.numAgents)])/sum(centralities_as_list)

        for question_num in range(self.influenceMatrixIndex*5 ,(self.influenceMatrixIndex+1)*5):
            self.centralities[question_num] = centralities_as_list + agent_weighted_centrality_perf

        #Move to next influence matrix
        self.influenceMatrixIndex+=1

        return np.asarray(W1.todense()), np.asarray(eig_centrality1), aperiodic, connected


    def updateCentrality1(self, influenceMatrixIndex):
        #Compute Eigen Vector Centrality for Humans
        graph = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph.add_edge(i,j,weight=self.memberInfluences[influenceMatrixIndex][i][j]/100)

        human_centralities = nx.eigenvector_centrality(graph, weight="weight")

        #Compute expected performance for machines

        """
        for i in range(0,self.teamSize):
            numerator = 0
            denominator = 0
            for j in range(0,self.teamSize):
                numerator+= self.centralities[j] * self.agentRatings[influenceMatrixIndex][j][i]
                denominator+= self.centralities[j]
            self.centralities.update({self.teamSize+i:numerator/denominator})

        """
        #Check that we have the correct total influence
        for i in range(self.teamSize):
            assert(sum(self.memberInfluences[influenceMatrixIndex][i][j] for j in range(self.numAgents)) == 100)

        #Make a probability

        agent_weighted_centrality_perf = [None for _ in range(self.numAgents)]
        '''
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([self.memberInfluences[influenceMatrixIndex][i][j]/100. for j in range(self.numAgents)])
        '''
        centralities_as_list = [value for value in human_centralities.values()]
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([centralities_as_list[j]*self.agentRatings[influenceMatrixIndex][j][i] for j in range(self.numAgents)])/sum(centralities_as_list)

        self.centralities = [[] for _ in range(self.numQuestions)]
        for question_num in range(self.influenceMatrixIndex*5 ,(self.influenceMatrixIndex+1)*5):
            self.centralities[question_num] = centralities_as_list + agent_weighted_centrality_perf

        #Move to next influence matrix
        self.influenceMatrixIndex+=1


    def calculatePerformanceProbability(self, questionNumber, idx):
        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize)]

        person_response = []

        for i in range(0,self.teamSize):
            #print("i:")
            #print(i)
            #print(self.centralities[idx][i])
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            # print(individulResponse)
            person_response.append(individulResponse)
            index = self.options[idx].index(individulResponse)
            #print("index:")
            #print(index)
            #print(self.centralities[idx][i])
            probabilities[index] += self.centralities[idx][i]
        #print("probability:")
        #print(probabilities)
        # Normalize the probabilties
        totalProbability = sum(probabilities)
        probabilities[:] = [x / totalProbability for x in probabilities]


        prob_agent = [0,0,0,0,0,0,0,0]
        # Add expected Performance for Agents
        for i in range(0, self.numAgents):
            #which agents should have a centrality of 1?
            if self.centralities[idx][self.getAgentForHuman(i)] == 1:
                # probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
                prob_agent[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]

            #which agents should have a positive centrality
            elif self.centralities[idx][i+self.teamSize] >= 0:
                # probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
                prob_agent[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            else:
                assert(False) # no negative centralities allowed

        four_agent_prob = prob_agent[4:]
        l1 = 0
        for i in range(4):
            l1 += four_agent_prob[i]

        for i in range(4):
            four_agent_prob[i] /= l1 

        ap = 1
        for agt_prob in four_agent_prob:
            ap = ap * (1-agt_prob)

        ap = 1 - ap

        probabilities = probabilities + [ap]
        return probabilities, person_response


    def calculateModelAccuracy(self,perQuestionRewards,probabilities,idx):
        highestRewardOption = max(perQuestionRewards[0:4])
        highestRewardAgent = max(perQuestionRewards[4:8])
        modelAccuracy = 0
        count = 0

        if highestRewardOption >= highestRewardAgent:
            for i in range(0,self.teamSize):
                if highestRewardOption == perQuestionRewards[i] and self.options[idx][i]==self.correctAnswers[idx]:
                    count+=1
                    modelAccuracy = 1
            modelAccuracy = modelAccuracy * count / (perQuestionRewards[0:4].count(highestRewardOption))
        else:
            for i in range(self.teamSize,self.teamSize*2):
                if highestRewardAgent == perQuestionRewards[i]:
                    modelAccuracy += probabilities[i] * (perQuestionRewards[4:8].count(highestRewardAgent))
        return modelAccuracy

    # Expected rewards for (all options + all agents)
    def calculateExpectedReward(self, probabilities):
        perQuestionRewards = list()
        for j in range(0,self.teamSize):
            perQuestionRewards.append(self.c*probabilities[j] + (self.z)*(1-probabilities[j]))

        # for j in range(0,self.teamSize):
        #     perQuestionRewards.append((self.c+self.e)*probabilities[self.getAgentForHuman(j)] + (self.z+self.e)*(1-probabilities[self.getAgentForHuman(j)]))
        perQuestionRewards.append((self.c+self.e)*probabilities[-1] + (self.z+self.e)*(1-probabilities[-1]))

        # print("probabilities: ")
        # print(probabilities)
        # print("perQuestionRewards: ")
        # print(perQuestionRewards)
        return perQuestionRewards

    def computePerformance(self):

        probabilitiesNB1 = list()
        probabilitiesRANDOM = list()
        group_accuracies = list()
        expectedP = list()
        
        random_p = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]

        # Compute Reward for NB1

        rewardsNB1 = list()
        rewardsCENT1 = list()     

        self.alphas = [1,1,1,1,1,1,1,1]
        self.betas = np.ones(8, dtype = int)
        for i in range(0,self.numQuestions):
            expectedPerformance, all_probabilities, five_probabilities,human_accuracy, group_accuracy, machine_accuracy, machineAnswer = self.naiveProbability(self.questionNumbers[i],i)
            
            # probabilitiesNB1.append(all_probabilities)
            probabilitiesNB1.append(five_probabilities)
            probabilitiesRANDOM.append(random_p)

            # bb = np.exp(rewards)
            # omega = 0.9
            # for k in range(4):
            #     bb[k] *= omega
            # for k in range(4,8):
            #     bb[k] *= 1-omega
            # l1 = 0
            # for k in range(8):
            #     l1 += bb[k]
            # probabilitiesNB1.append([bb[k] / l1 for k in range(8)])
    
            rewards = self.calculateExpectedReward(five_probabilities)
            rewardsNB1.append(rewards)

            expectedP.append(expectedPerformance)

        
        #Compute Reward for CENT1 model
        probabilitiesCENT1 = list()
        #largest_components = []
        if_aperiodic = []
        if_connected = []
        adjacency = []
        eig_cen = []
        # self.influenceMatrixIndex = 0
        # self.centralities = [[] for _ in range(self.numQuestions)]
        for i in range(0,self.numCentralityReports):
            adj, eigenvector_centrality, aperiodic, connected = self.updateCentrality(self.influenceMatrixIndex)
            #largest_components.append(len(largest))
            if_aperiodic.append(aperiodic)
            if_connected.append(connected)
            adjacency.append(adj)
            eig_cen.append(eigenvector_centrality)

        before_consensus = []
        for i in range(0,self.numQuestions):
            probabilities, person_response = self.calculatePerformanceProbability(self.questionNumbers[i],i)
            if len(set(person_response)) == 1:
                before_consensus.append(1)
            else:
                before_consensus.append(0)
            probabilitiesCENT1.append(probabilities)
            rewards = self.calculateExpectedReward(probabilities)
            rewardsCENT1.append(rewards)

        return expectedP, rewardsNB1, rewardsCENT1, probabilitiesNB1,probabilitiesCENT1, probabilitiesRANDOM

    def naiveProbability1(self, questionNumber, idx, exploit):
        expectedPerformance = list()
        individualResponse = list()
        probabilities = list()
        human_accuracy = list()

        machine_accuracy = [None for _ in range(self.numAgents)]
        group_accuracy = 0

        #Save human expected performance based
        for i in range(0,self.teamSize):
            expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
            individualResponse.append(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())
            self.updateAlphaBeta(i,self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any(),self.correctAnswers[idx])

            ans = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            if ans == self.correctAnswers[idx]:
                human_accuracy.append(1)
            else:
                human_accuracy.append(0)

        if (self.groupSubmission["groupAnswer"].iloc[idx] == self.correctAnswers[idx]):
            group_accuracy = 1

        indxQ = -1
        anyMachineAsked = False
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            # print("k")
            # print(k)
            anyMachineAsked = True

        # Add expected Performance for Agents
        for i in range(self.teamSize, self.teamSize+self.numAgents):
            expectedPerformance.append(0.5 + 0.5 * beta.mean(self.alphas[i],self.betas[i]))
            # update alpha beta for that machine


        # print("expectedPerformance")
        # print(expectedPerformance)

        prob_class_responses = [0 for _ in range(len(self.options[idx]))]
        prob_resp_given_class = [0 for _ in range(len(self.options[idx]))]

        # #Update machine accuracy
        if exploit[idx] == 1 and anyMachineAsked:

            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
            self.updateAlphaBeta(self.getAgentForHuman(k), machineAnswer, self.correctAnswers[idx])
            self.machineUseCount[k]+=1

            if self.firstMachineUsage[k] == -1:
                self.firstMachineUsage[k] = idx

            if (machineAnswer == self.correctAnswers[idx]):
                machine_accuracy[k] = 1
            else:
                machine_accuracy[k] = 0

            # machine_accuracy[k] = 1

            #human-agent
            # Conditional Probability
            # Do a bayes update
            denominator = 0
            numerator = [1. for _ in range(len(self.options[idx]))]
            prob_class = 0.25
            prob_rsep = 0
            
            # for opt_num in range(0,len(self.options[idx])):
            #     numerator = prob_class

                # for person_num in range(0,self.teamSize):
                #     if individualResponse[person_num] == self.options[idx][opt_num]:
                #         numerator *= expectedPerformance[person_num]
                #     else:
                #         numerator *= (1 - expectedPerformance[person_num])/3
                #old used in the ICML paper
                # if machineAnswer == self.options[idx][opt_num]:
                #     numerator *= expectedPerformance[4+k]
                # else:
                #     numerator *= (1 - expectedPerformance[4+k])/3
                
                #new
                # agent_performance = expectedPerformance[4:]
                # a_wrong = 1
                # for ap in agent_performance:
                #     a_wrong = a_wrong * (1 - ap)
                # a_right = 1 - a_wrong
                # if machineAnswer == self.options[idx][opt_num]:
                #     numerator *= a_right
                # else:
                #     numerator *= a_wrong



            #     prob_resp_given_class[opt_num] = numerator
            # prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]

            # #agent alone
            for opt_num in range(0,len(self.options[idx])):
                prob_class_responses[opt_num] = (1 - expectedPerformance[4+k])/3
            for opt_num in range(0,len(self.options[idx])):
                if machineAnswer == self.options[idx][opt_num]:
                    prob_class_responses[opt_num] = expectedPerformance[4+k]
                    break

            # #random
            # for opt_num in range(4):
            #     prob_class_responses[opt_num] = 0.25

        else:
            # Conditional Probability
            # human alone
            # Do a bayes update
            denominator = 0
            numerator = [1. for _ in range(len(self.options[idx]))]
            prob_class = 0.25
            prob_rsep = 0

            for opt_num in range(0,len(self.options[idx])):
                prob_resp = 0
                numerator = prob_class
                for person_num in range(0,self.teamSize):
                    if individualResponse[person_num] == self.options[idx][opt_num]:
                        numerator *= expectedPerformance[person_num]
                    else:
                        numerator *= (1 - expectedPerformance[person_num])/3
                    prob_resp += numerator
                prob_resp_given_class[opt_num] = numerator
            prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]
            # print("prob_class_responses: ")
            # print(prob_class_responses)

        #ANSIs this updating agent probabilities?
        l1 = 0
        for i in range(self.teamSize):
            l1 += expectedPerformance[self.teamSize+i]

        for i in range(self.teamSize):
            probabilities.append(1.0 * expectedPerformance[self.teamSize+i]/l1)

        #8 probability values returned
        # first set is for options (sums to 1)

        assert(sum(prob_class_responses) > 0.999 and sum(prob_class_responses) < 1.001)
        #second set is for machines
        # prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)]/l1 for k in range(self.teamSize)]

        prob_all_class_responses = prob_class_responses
        return expectedPerformance, prob_all_class_responses,human_accuracy,group_accuracy,machine_accuracy


    # def naiveProbability1(self, questionNumber, idx, exploit):
    #     expectedPerformance = list()
    #     individualResponse = list()
    #     probabilities = list()
    #     human_accuracy = list()

    #     machine_accuracy = [None for _ in range(self.numAgents)]
    #     group_accuracy = 0

    #     #Save human expected performance based
    #     for i in range(0,self.teamSize):
    #         expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
    #         individualResponse.append(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())
    #         self.updateAlphaBeta(i,self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any(),self.correctAnswers[idx])

    #         ans = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
    #         if ans == self.correctAnswers[idx]:
    #             human_accuracy.append(1)
    #         else:
    #             human_accuracy.append(0)

    #     if (self.groupSubmission["groupAnswer"].iloc[idx] == self.correctAnswers[idx]):
    #         group_accuracy = 1

    #     indxQ = -1
    #     anyMachineAsked = False
    #     if(int(float(questionNumber)) in self.machineAskedQuestions):
    #         indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
    #         sender = self.machineAsked['sender'].iloc[indxQ]
    #         k = self.teamArray.index(sender)
    #         # print("k")
    #         # print(k)
    #         anyMachineAsked = True

    #     # Add expected Performance for Agents
    #     for i in range(self.teamSize, self.teamSize+self.numAgents):
    #         expectedPerformance.append(0.5 + 0.5 * beta.mean(self.alphas[i],self.betas[i]))
    #         # update alpha beta for that machine


    #     # print("expectedPerformance")
    #     # print(expectedPerformance)

    #     prob_class_responses_HA = [0 for _ in range(len(self.options[idx]))]
    #     prob_class_responses_H = [0 for _ in range(len(self.options[idx]))]
    #     prob_class_responses_A = [0 for _ in range(len(self.options[idx]))]
        

    #     # human alone
    #     # Do a bayes update
    #     prob_resp_given_class = [0 for _ in range(len(self.options[idx]))]
    #     denominator = 0
    #     numerator = [1. for _ in range(len(self.options[idx]))]
    #     prob_class = 0.25
    #     prob_rsep = 0

    #     for opt_num in range(0,len(self.options[idx])):
    #         prob_resp = 0
    #         numerator = prob_class
    #         for person_num in range(0,self.teamSize):
    #             if individualResponse[person_num] == self.options[idx][opt_num]:
    #                 numerator *= expectedPerformance[person_num]
    #             else:
    #                 numerator *= (1 - expectedPerformance[person_num])/3
    #             prob_resp += numerator
    #         prob_resp_given_class[opt_num] = numerator
    #     prob_class_responses_H = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]
        
    #     # #Update machine accuracy
    #     prob_resp_given_class = [0 for _ in range(len(self.options[idx]))]
    #     if anyMachineAsked:
    #         machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
    #         self.updateAlphaBeta(self.getAgentForHuman(k), machineAnswer, self.correctAnswers[idx])
    #         self.machineUseCount[k]+=1

    #         if self.firstMachineUsage[k] == -1:
    #             self.firstMachineUsage[k] = idx

    #         if (machineAnswer == self.correctAnswers[idx]):
    #             machine_accuracy[k] = 1
    #         else:
    #             machine_accuracy[k] = 0

    #         # machine_accuracy[k] = 1

    #         #agent alone
    #         for opt_num in range(0,len(self.options[idx])):
    #             prob_class_responses_A[opt_num] = (1 - expectedPerformance[4+k])/3
    #         for opt_num in range(0,len(self.options[idx])):
    #             if machineAnswer == self.options[idx][opt_num]:
    #                 prob_class_responses_A[opt_num] = expectedPerformance[4+k]
    #                 break

    #         if exploit[idx] == 1:
    #             denominator = 0
    #             numerator = [1. for _ in range(len(self.options[idx]))]
    #             prob_class = 0.25
    #             prob_rsep = 0
    #             for opt_num in range(0,len(self.options[idx])):
    #                 numerator = prob_class

    #                 for person_num in range(0,self.teamSize):
    #                     if individualResponse[person_num] == self.options[idx][opt_num]:
    #                         numerator *= expectedPerformance[person_num]
    #                     else:
    #                         numerator *= (1 - expectedPerformance[person_num])/3
    #                 #old
    #                 if machineAnswer == self.options[idx][opt_num]:
    #                     numerator *= expectedPerformance[4+k]
    #                 else:
    #                     numerator *= (1 - expectedPerformance[4+k])/3
    #                 #new
    #                 # agent_performance = expectedPerformance[4:]
    #                 # a_wrong = 1
    #                 # for ap in agent_performance:
    #                 #     a_wrong = a_wrong * (1 - ap)
    #                 # a_right = 1 - a_wrong
    #                 # if machineAnswer == self.options[idx][opt_num]:
    #                 #     numerator *= a_right
    #                 # else:
    #                 #     numerator *= a_wrong

    #                 prob_resp_given_class[opt_num] = numerator
    #             prob_class_responses_HA = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]

    #         # #agent alone
    #         # for opt_num in range(0,len(self.options[idx])):
    #         #     prob_class_responses[opt_num] = (1 - expectedPerformance[4+k])/3
    #         # for opt_num in range(0,len(self.options[idx])):
    #         #     if machineAnswer == self.options[idx][opt_num]:
    #         #         prob_class_responses[opt_num] = expectedPerformance[4+k]
    #         #         break

    #         # #random
    #         # for opt_num in range(4):
    #         #     prob_class_responses[opt_num] = 0.25

    #         else:
    #             # Conditional Probability
    #             # human alone
    #             # Do a bayes update
    #             denominator = 0
    #             numerator = [1. for _ in range(len(self.options[idx]))]
    #             prob_class = 0.25
    #             prob_rsep = 0

    #             for opt_num in range(0,len(self.options[idx])):
    #                 prob_resp = 0
    #                 numerator = prob_class
    #                 for person_num in range(0,self.teamSize):
    #                     if individualResponse[person_num] == self.options[idx][opt_num]:
    #                         numerator *= expectedPerformance[person_num]
    #                     else:
    #                         numerator *= (1 - expectedPerformance[person_num])/3
    #                     prob_resp += numerator
    #                 prob_resp_given_class[opt_num] = numerator
    #             prob_class_responses_HA = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]
    #         # print("prob_class_responses: ")
    #         # print(prob_class_responses)

    #     #ANSIs this updating agent probabilities?
    #     l1 = 0
    #     for i in range(self.teamSize):
    #         l1 += expectedPerformance[self.teamSize+i]

    #     for i in range(self.teamSize):
    #         probabilities.append(1.0 * expectedPerformance[self.teamSize+i]/l1)

    #     #8 probability values returned
    #     # first set is for options (sums to 1)

    #     assert(sum(prob_class_responses_HA) > 0.999 and sum(prob_class_responses_HA) < 1.001)
    #     assert(sum(prob_class_responses_H) > 0.999 and sum(prob_class_responses_H) < 1.001)
    #     assert(sum(prob_class_responses_A) > 0.999 and sum(prob_class_responses_A) < 1.001)
    #     #second set is for machines
    #     # prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)]/l1 for k in range(self.teamSize)]

    #     return expectedPerformance, prob_class_responses_HA,prob_class_responses_H,prob_class_responses_A,human_accuracy,group_accuracy,machine_accuracy

    def calculateRewards_NB(self,exploit):
        rewardsNB1 = list()
        probabilitiesNB1 = list()
        # probabilitiesNB1_H = list()
        # probabilitiesNB1_A = list()
        group_accuracies = list()
        group_accuracy_per_question = list() # for each question
        expectedP = list()
        h1 = list()
        h2 = list()
        h3 = list()
        h4 = list()
        
        # Compute Reward for NB1
        self.alphas = [1,1,1,1,1,1,1,1]
        self.betas = np.ones(8, dtype = int)
        for i in range(0,self.numQuestions):

            expectedPerformance, all_probabilities, human_accuracy, group_accuracy, machine_accuracy = self.naiveProbability1(self.questionNumbers[i],i,exploit)
            group_accuracy_per_question.append(group_accuracy)
            expectedP.append(expectedPerformance)
            h1.append(human_accuracy[0])
            h2.append(human_accuracy[1])
            h3.append(human_accuracy[2])
            h4.append(human_accuracy[3])
            probabilitiesNB1.append(all_probabilities)
            # probabilitiesNB1_H.append(H_probabilities)
            # probabilitiesNB1_A.append(A_probabilities)
            # rewardsNB1.append(self.calculateExpectedReward(all_probabilities))
            # print("expectedPerformance: ")
            # print(expectedPerformance)
            # print("all_probabilities: ")
            # print(all_probabilities)
            # print("human_accuracy: ")
            # print(human_accuracy)
            # print("machine_accuracy: ")
            # print(machine_accuracy)
        # print(rewardsNB1)

        #Compute Reward for CENT1 model
        rewardsCENT1 = list()
        probabilitiesCENT1 = list()

        # for i in range(0,self.numCentralityReports):
        #     self.updateCentrality1(self.influenceMatrixIndex)

        # print("centrality: ")
        # print(self.centralities)

        # before_consensus = []
        # for i in range(0,self.numQuestions):
        #     # print("question number:")
        #     # print(i)
        #     probabilities, person_response = self.calculatePerformanceProbability1(self.questionNumbers[i],i)
        #     if len(set(person_response)) == 1:
        #         before_consensus.append(1)
        #     else:
        #         before_consensus.append(0)

        #     probabilitiesCENT1.append(probabilities)
            # rewardsCENT1.append(self.calculateExpectedReward(probabilities))

        # print("before consensus:")
        # print(before_consensus)

        best_human_accuracy = max([sum(h1)/len(h1), sum(h2)/len(h2), sum(h3)/len(h3), sum(h4)/len(h4)])
        return expectedP, rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1

    def calculatePerformanceProbability_CENT(self, questionNumber, idx, be, exploit):

        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize + self.numAgents)]

        # print("questionNumber")
        # print(questionNumber)
        # print("idx")
        # print(idx)

        for i in range(0,self.teamSize):
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            # print("individulResponse")
            # print(individulResponse)
            index = self.options[idx].index(individulResponse)
            # print("options: ")
            # print(self.options[idx])
            # print("index")
            # print(index)
            # print(self.centralities)
            probabilities[index] += self.centralities[idx][i]

        # Normalize the probabilties
        # print("probability:")
        # print(probabilities)
        totalProbability = sum(probabilities)
        probabilities[:] = [x / totalProbability for x in probabilities]

        # Add expected Performance for Agents
        for i in range(0, self.numAgents):
            #which agents should have a centrality of 1?
            if self.centralities[idx][self.getAgentForHuman(i)] == 1:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            #which agents should have a positive centrality
            elif self.centralities[idx][i+self.teamSize] >= 0:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            else:
                assert(False) # no negative centralities allowed

        # print("probabilities: ")
        # print(probabilities)

        # l1 = 0
        # for i in range(4,8):
        #     l1 += probabilities[i]

        # for i in range(4,8):
        #     probabilities[i] /= l1 




        # print("probability1:")
        # print(probabilities)

        indxQ = -1
        anyMachineAsked = False
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            # print("k")
            # print(k)
            anyMachineAsked = True

        if exploit[idx] == 1 and anyMachineAsked:
            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
            ind = self.options[idx].index(machineAnswer)

            #cent-agent
            # for i in range(0,4):
            #    probabilities[i] *= (1-be)
            # probabilities[ind] += probabilities[4+k] * be

            # probabilities[ind] += probabilities[4+k]

            # l1 = 0
            # for i in range(0,4):
            #     l1 += probabilities[i]

            # for i in range(0,4):
            #     probabilities[i] /= l1

            # print("probability2:")
            # print(probabilities)

            # #agent
            for i in range(0,4):
                probabilities[i] = 1.0 * (1-probabilities[4+k]) / 3
            probabilities[ind] = probabilities[4+k]

            l1 = 0
            for i in range(0,4):
                l1 += probabilities[i]

            for i in range(0,4):
                probabilities[i] /= l1

            # #random
            # for i in range(0,4):
            #     probabilities[i] = 0.25

        else:
            l1 = 0
            for i in range(0,4):
                l1 += probabilities[i]

            for i in range(0,4):
                probabilities[i] /= l1 


        return probabilities


    def calculateRewards_CENT(self,be,exploit):
        rewardsNB1 = list()
        probabilitiesNB1 = list()
        group_accuracies = list()
        group_accuracy_per_question = list() # for each question
        expectedP = list()
        h1 = list()
        h2 = list()
        h3 = list()
        h4 = list()
        
        # Compute Reward for NB1
        # for i in range(0,self.numQuestions):
        #     expectedPerformance, all_probabilities, human_accuracy, group_accuracy, machine_accuracy = self.naiveProbability(self.questionNumbers[i],i)
        #     group_accuracy_per_question.append(group_accuracy)
        #     expectedP.append(expectedPerformance)
        #     h1.append(human_accuracy[0])
        #     h2.append(human_accuracy[1])
        #     h3.append(human_accuracy[2])
        #     h4.append(human_accuracy[3])
        #     probabilitiesNB1.append(all_probabilities)
        #     rewardsNB1.append(self.calculateExpectedReward(all_probabilities))
            # print("expectedPerformance: ")
            # print(expectedPerformance)
            # print("all_probabilities: ")
            # print(all_probabilities)
            # print("human_accuracy: ")
            # print(human_accuracy)
            # print("machine_accuracy: ")
            # print(machine_accuracy)
        # print(rewardsNB1)

        #Compute Reward for CENT1 model
        rewardsCENT1 = list()
        probabilitiesCENT1 = list()
        for i in range(0,self.numCentralityReports):
            adj, eigenvector_centrality, aperiodic, connected = self.updateCentrality(self.influenceMatrixIndex)

        for i in range(0,self.numQuestions):
            probabilities = self.calculatePerformanceProbability_CENT(self.questionNumbers[i],i,be,exploit)
            probabilitiesCENT1.append(probabilities)
            rewardsCENT1.append(self.calculateExpectedReward(probabilities))

        # best_human_accuracy = max([sum(h1)/len(h1), sum(h2)/len(h2), sum(h3)/len(h3), sum(h4)/len(h4)])
        best_human_accuracy = 1
        return expectedP, rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1, group_accuracy_per_question, best_human_accuracy


    #--Deprecated--
    def computePTaccuracy(self, pi):
        PTrewards = list()
        for i in range(0,len(pi)):
            PTrewards.append(model.calculateExpectedReward(pi[i]))
        accuracy = list()
        for i in range(0,len(pi)):
            if i==0:
                accuracy.append(self. calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize))/(i+1))
            else:
                accuracy.append((self.calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize)) + (i*accuracy[i-1]))/(i+1))
        return PTrewards, accuracy

    def softmax(self, vec):
        return np.exp(vec) / np.sum(np.exp(vec), axis=0)

        # Called in loss function --Deprecated--
    def newValues(self,values):
        least = min(values)
        values[:] = [i-least for i in values]
        values[:] = [i/sum(values) for i in values]
        return values

    def computeActionTaken(self):
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i] == "Consensus Not Reached":
                self.actionTaken.append(-1)
            # elif int(float(self.questionNumbers[i])) in self.exploreQuestions:
            elif int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
            #     # self.actionTaken.append(self.teamSize + np.where(self.machineUsed[i] == True)[0][0])
                self.actionTaken.append(4)
            else:
                self.actionTaken.append(self.options[i].index(self.groupSubmission.groupAnswer[i]))

            if self.groupSubmission.groupAnswer[i] == "Consensus Not Reached":
                self.actionTaken1.append(-1)
            else:
                self.actionTaken1.append(self.options[i].index(self.groupSubmission.groupAnswer[i]))

#     Computes V1 to V8 for a given question --Deprecated--
    def computeCPT(self,alpha,gamma,probabilities):
        values = list()
        for i in range(0,2*self.teamSize):
            if i<4:
                values.append((math.pow(self.c, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
            else:
                values.append((math.pow(self.c+self.z, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z + self.e), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
        return values

    #--Deprecated--
    def bestAlternative(self,values,action):
        highest = max(values)
        if highest!=action:
            return highest
        temp = list(filter(lambda a: a != highest, values))
        if len(temp)==0:
            return -100
        return max(temp)

#     Compute P_I for CPT models --Deprecated--
    def computePI(self, values, actionTaken,lossType):
        z = self.bestAlternative(values,values[actionTaken])
        if (z==-100):
            if lossType=="logit":
                return 0.25
            else:
                return 0
        z = values[actionTaken]-z
        if lossType=="softmax":
            return z
        return 1/(1+math.exp(-z))


    #action in 0,...,numAgents
    def computeLoss(self,params,probabilities,theta, chosen_action,lossType,loss_function,modelName):

        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)

        # prob1 = probabilities[0:4]
        # prob1 = np.sort(prob1)
        # diff = prob1[3]-prob1[2]

        # exploit = 0

        # RAND = random.uniform(0, 1)
        # if diff <= 0.3 or (diff > 0.3 and RAND <= max(prob1) * theta):
        #     exploit = 1
        #     probabilities[0] = 0.0001
        #     probabilities[1] = 0.0001
        #     probabilities[2] = 0.0001
        #     probabilities[3] = 0.0001
        #     probabilities[4] = 0.9996
        # num = 5

        # prob1 = probabilities[0:4]
        # prob1 = np.sort(prob1)
        # diff = prob1[3]-prob1[2]

        # exploit = 0

        # RAND = random.uniform(0, 1)
        # if diff <= 0.3 or (diff > 0.3 and RAND <= max(prob1) * theta):
        #     exploit = 1
        # else:
        #     probabilities[0] = 0.0001
        #     probabilities[1] = 0.0001
        #     probabilities[2] = 0.0001
        #     probabilities[3] = 0.0001
        #     probabilities[4] = 0.9996
        # num = 5

        prospects= []
        for probability in probabilities[0:self.teamSize]:
            prospectSuccess = self.c, probability
            prospectFailure = self.z, 1-probability
            prospects.append((prospectSuccess,prospectFailure))

        # print("prospects:")
        # print(prospects)

        cpt_vals_option = evaluateProspectVals(params,prospects)

        prospects= []
        for probability in probabilities[self.teamSize:]:
            prospectSuccess = self.c +self.e, probability
            prospectFailure = self.z +self.e, 1-probability
            prospects.append((prospectSuccess,prospectFailure))

        # print("probabilities: ")
        # print(probabilities)
        cpt_vals_agent = evaluateProspectVals(params,prospects)

        cpt_vals = []
        for r in cpt_vals_option:
            cpt_vals.append(r)

        for r in cpt_vals_agent:
            cpt_vals.append(r)

        ground = chosen_action

        soft_prob = self.softmax(cpt_vals)
        # print("soft_prob")
        # print(soft_prob)
        predicted = np.argmax(soft_prob)

        prob1 = probabilities[0:4]
        prob1 = np.sort(prob1)
        diff = prob1[3]-prob1[2]

        exploit = 0

        RAND = random.uniform(0, 1)
        threshold = 0.6
        # if diff <= threshold and RAND <= max(prob1) * theta:
        # if diff <= threshold and RAND <= theta:
        if diff <= threshold or (diff > threshold and RAND <= max(prob1) * theta):
            exploit = 1
        else:
            predicted = 4
            soft_prob = np.zeros(5)
            soft_prob[0] = 0.0001
            soft_prob[1] = 0.0001
            soft_prob[2] = 0.0001
            soft_prob[3] = 0.0001
            soft_prob[4] = 0.9996
        num = 5

        # if diff <= 0.3 or (diff > 0.3 and RAND <= max(prob1) * theta):
        #     exploit = 1
        #     predicted = 4
        #     soft_prob = np.zeros(5)
        #     soft_prob[0] = 0.0001
        #     soft_prob[1] = 0.0001
        #     soft_prob[2] = 0.0001
        #     soft_prob[3] = 0.0001
        #     soft_prob[4] = 0.9996 
        # num = 5
        


        # print("predicted: ")
        # print(predicted)
        # print("ground: ")
        # print(chosen_action)
        acc = 0
        if predicted == ground:
            acc = 1

        loss = 0

        if loss_function == "binary":
            arg = soft_prob[chosen_action]
            if np.isnan(arg):
                arg = 1.
            if arg == 0:
                arg += 0.0000001
            loss = -1. * math.log(arg)

        elif loss_function == "variational_H_qp":

            i = np.argmax(soft_prob)
            j = chosen_action
            # print(soft_prob)

            if i == j:
                for k in range(num):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                index1 = []
                index2 = []
                for k in range(num):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                m = 0
                for ind in index1:
                    m += soft_prob[ind]
                
                variation_q =  np.zeros(num)
                for ind in index1:
                    variation_q[ind] = 1.0 * m / len(index1)

                for ind in index2:
                    variation_q[ind] = soft_prob[ind]



                # variation_q =  np.zeros(8)
                # alpha_star = 0.5 * (soft_prob[i] + soft_prob[j])
                # variation_q[i] = alpha_star
                # variation_q[j] = alpha_star
                # index1 = []
                # index2 = []
                # index1.append(i)
                # index1.append(j)
                # for k in range(8):
                #     index2.append(int(k))
                # index3 = list(set(index2) - set(index1))
                
                # max_ind = j
                # for k in index3:
                #     variation_q[k] = soft_prob[k]
                #     if variation_q[k] > alpha_star:
                #         max_ind = k

                # if max_ind != j:
                #     s = 0
                #     for k in index3:
                #         s += variation_q[k]
                #     ratio = 1.0 * variation_q[max_ind] / s
                #     beta = 1.0 * (2 * ratio - 2.0 * alpha_star / (1 - 2 * alpha_star)) / (2 * ratio + 1)
                #     variation_q[i] = 1.0 * ratio / (2 * ratio + 1)
                #     variation_q[j] = 1.0 * ratio / (2 * ratio + 1)
                #     for k in index3:
                #         variation_q[k] *= (1 - beta)
                    # print("sum  of q:")
                    # print(variation_q)
                    # print(np.sum(variation_q))

                # print("soft_prob: ")
                # print(soft_prob)
                # print("variation_q: ")
                # print(variation_q)
                # print(sum(variation_q))
                for k in range(num):
                    loss += -soft_prob[k] * math.log(variation_q[k])
                # print("loss")
                # print(loss)
        else:
            i = np.argmax(soft_prob)
            j = chosen_action
            if i == j:
                for k in range(num):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:

                index1 = []
                index2 = []
                for k in range(num):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                
                variation_q =  np.zeros(num)
                for ind in index1:
                    variation_q[ind] = 1.0  / len(index1)


                # # print("predicted:")
                # # print(i)
                # # print("chosem_action:")
                # # print(j)
                # variation_q =  np.zeros(8)
                # index1 = []
                # index2 = []
                # index1.append(i)
                # index1.append(j)
                # for k in range(8):
                #     index2.append(int(k))
                # index3 = list(set(index2) - set(index1))

                # l = np.sqrt(soft_prob[i] * soft_prob[j])
                # index = []
                # for k in index3:
                #     if soft_prob[k] > l:
                #         index.append(k)

                # if len(index) == 0:
                #     variation_q[i] = 0.5
                #     variation_q[j] = 0.5

                # else:
                #     variation_q[i] = 1. / (len(index) + 2)
                #     variation_q[j] = 1. / (len(index) + 2)
                #     for kk in range(len(index)):
                #         variation_q[index[kk]] = 1. / (len(index) + 2)
                # # print("soft_prob: ")
                # # print(soft_prob)
                #print("variation_q: ")
                #print(variation_q)
                #print(sum(variation_q))

                for k in range(num):
                    loss += -variation_q[k] * math.log(soft_prob[k])
                    #if variation_q[k] == 0:
                    #    variation_q[k] = 0.00000001
                    #loss += variation_q[k] * math.log(variation_q[k])


        return loss, acc, predicted, ground, soft_prob, exploit


    def computeCPTLoss(self,params,probabilities,theta,lossType,loss_function,modelName):
        total_loss = 0
        per_question_loss = [None for _ in range(self.numQuestions)]
        per_option_loss = [None for _ in range(self.numQuestions)]
        per_agent_loss = [None for _ in range(self.numQuestions)]
        per_question_acc = [None for _ in range(self.numQuestions)]

        per_question_ground = [None for _ in range(self.numQuestions)]
        per_question_predicted = [None for _ in range(self.numQuestions)]
        per_question_exploit = [None for _ in range(self.numQuestions)]

        length = len(probabilities)
        new_prob = []


        start = 0
        if length==self.testSetSize:
            start = self.trainingSetSize
        
        q_ind = np.zeros(length)
        for question_num in range(length):
            #Here - How to handle consensus not reached case
            if self.actionTaken[start+question_num]==-1:
                # print("consensus not reached:")
                # print(question_num)
                q_ind[question_num] = 1
                continue

            assert(self.actionTaken[start+question_num] in range(self.teamSize+self.numAgents))
            loss, acc, predicted, ground, soft_prob, exploit = self.computeLoss(params,probabilities[question_num],theta,self.actionTaken[start+question_num],lossType,loss_function,modelName)
            new_prob.append(soft_prob)
            
            if ground < 4:
                per_option_loss[start+question_num] = loss
            else:
                per_agent_loss[start+question_num] = loss
            per_question_loss[start+question_num] = loss
            per_question_acc[start+question_num] = acc
            per_question_ground[start+question_num] = ground
            per_question_predicted[start+question_num] = predicted
            per_question_exploit[start+question_num] = exploit

        return per_question_loss, per_option_loss, per_agent_loss, per_question_acc, per_question_ground,per_question_predicted,per_question_exploit,new_prob,q_ind


    def computeAverageLossPerTeam(self,params, probabilities, theta, lossType, loss_function, modelName):

        if modelName == "random":

            per_question_loss = [None for _ in range(self.numQuestions)]
            per_question_acc = [None for _ in range(self.numQuestions)]

            length = len(probabilities)
            start = 0
            if length==self.testSetSize:
                start = self.trainingSetSize

            for question_num in range(length):
                #Here - How to handle consensus not reached case
                if self.actionTaken[start+question_num]==-1:
                    continue
                
                assert(self.actionTaken[start+question_num] in range(self.teamSize+self.numAgents))
                prob = 1.0/5
                per_question_loss[start+question_num] = -1.0*math.log(prob)
                per_question_acc[start+question_num] = prob
                ground = [None for _ in range(8)]
                predicted = [None for _ in range(8)]
                q_ind = []

                # per_question_loss, per_question_acc, ground,predicted = self.computeCPTLoss(params,probabilities,lossType,loss_function,'nb')

        else:

            per_question_loss, per_option_loss, per_agent_loss, per_question_acc, ground,predicted,exploit,new_prob,q_ind = self.computeCPTLoss(params,probabilities,theta,lossType,loss_function,modelName)

        # print("per_question_loss:")
        # print(per_question_loss)
        total_loss = 0
        all_loss = []
        count = 0
        for loss in per_question_loss:
            if (loss != None):
                all_loss.append(loss)
                total_loss += loss
                count += 1
            else:
                all_loss.append(-1)

        if count!=0:
            total_loss /= count

        total_option_loss = 0
        all_loss_option = []
        count = 0
        for loss in per_option_loss:
            if (loss != None):
                all_loss_option.append(loss)
                total_option_loss += loss
                count += 1
            else:
                all_loss_option.append(-1)

        if count!=0:
            total_option_loss /= count

        total_agent_loss = 0
        all_loss_agent = []
        count = 0
        for loss in per_agent_loss:
            if (loss != None):
                all_loss_agent.append(loss)
                total_agent_loss += loss
                count += 1

        if count!=0:
            total_agent_loss /= count


        total_acc = 0
        count_acc = 0
        for acc in per_question_acc:
            if (acc != None):
                total_acc += acc
                count_acc += 1

        if count_acc!=0:
            total_acc /= count_acc

        # ground1 = []
        # predicted1 = []
        # for ele in ground:
        #     if ele != None:
        #         ground1.append(ele)

        # for ele in predicted:
        #     if ele != None:
        #         predicted1.append(ele)

        # print("ground1: ")
        # print(ground1)
        # print("predicted1: ")
        # print(predicted1)

        return all_loss, all_loss_option, all_loss_agent, total_loss, total_option_loss, total_agent_loss, total_acc, ground,predicted,exploit,new_prob,q_ind


    def chooseCPTParameters(self, probabilities,theta,lossType,loss_function,modelName):
        # hAlpha, hGamma,hLambda =  (None,None,None)
        Alpha, Beta, Lambda, GammaGain, GammaLoss =  (None,None,None,None,None)

        Loss = np.float("Inf")

        for alpha in np.arange(0,1.1,0.1):
            for lamda in np.arange(1,11,1):
                for gammaGain in np.arange(0,1.1,0.1):
                    for gammaLoss in np.arange(0,1.1,0.1):

                        all_loss, all_loss_option, all_loss_agent, loss_cpt, option_loss_cpt, agent_loss_cpt, acc_cpt, ground,predicted,exploit,new_prob,q_ind = self.computeAverageLossPerTeam((alpha,alpha,lamda,gammaGain,gammaLoss),probabilities,theta,lossType,loss_function,modelName)

                        if (loss_cpt<Loss):
                            Loss = loss_cpt
                            Alpha = alpha
                            Beta = alpha
                            Lambda = lamda
                            GammaGain = gammaGain
                            GammaLoss = gammaLoss

                        # if (agent_loss_cpt<Loss):
                        #     Loss = agent_loss_cpt
                        #     Alpha = alpha
                        #     Beta = alpha
                        #     Lambda = lamda
                        #     GammaGain = gammaGain
                        #     GammaLoss = gammaLoss

        assert(Alpha != None)
        assert(Beta != None)
        assert(Lambda != None)
        assert(GammaGain != None)
        assert(GammaLoss != None)

        return (Alpha, Beta, Lambda, GammaGain, GammaLoss)


    def randomModel(self):
        prAgent = len(self.machineAskedQuestions)/self.numQuestions
        prHuman = 1.0-prAgent
        qi = list()
        for i in range(self.trainingSetSize,self.numQuestions):
            temp = [0.25*prHuman for j in range(0,self.teamSize)]
            for j in range(0,self.teamSize):
                temp.append(0.25*prAgent)
            qi.append(temp)
        return qi

    # Agent i has agent i + teamSize
    def getAgentForHuman(self, k):
        return self.teamSize + k

    def computeHumanLoss(self,params,probabilities,chosen_action,lossType,loss_function,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)
        prospects= []

        # print("probabilities")
        # print(probabilities)
        for probability in probabilities:
            prospectSuccess = self.c, probability
            prospectFailure = self.z, 1-probability
            prospects.append((prospectSuccess,prospectFailure))
        # print("human: ")
        # print(prospects)

        cpt_vals = evaluateProspectVals(params,prospects)
        # print(cpt_vals)
        # print(chosen_action)
        soft_prob = self.softmax(cpt_vals)
        # print("human_prob:")
        # print(soft_prob)
        predicted = np.argmax(soft_prob)
        ground = chosen_action
        acc = 0
        if predicted == ground:
            acc = 1

        loss = 0
        if loss_function == "binary":
            arg = soft_prob[chosen_action]
            # print(arg)
            if np.isnan(arg):
                arg = 1.
            if arg == 0:
                arg += 0.0000001
            loss = -1. * math.log(arg)
        elif loss_function == "variational_H_qp":
            
            i = np.argmax(soft_prob)
            j = chosen_action
            # print(soft_prob)

            if i == j:
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                m = 0
                for ind in index1:
                    m += soft_prob[ind]
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0 * m / len(index1)

                for ind in index2:
                    variation_q[ind] = soft_prob[ind]

                for k in range(4):
                    loss += -soft_prob[k] * math.log(variation_q[k])
        else:
            i = np.argmax(soft_prob)
            j = chosen_action
            if i == j:
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:

                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0  / len(index1)

                for k in range(4):
                    loss += -variation_q[k] * math.log(soft_prob[k])

        return loss, acc, predicted, ground

    def computeCPTLoss1(self,params,probabilities,label,lossType,loss_function,modelName):
        total_loss = 0
        per_question_agent_loss = [None for _ in range(self.numQuestions)]
        per_question_option_loss = [None for _ in range(self.numQuestions)]
        per_question_ground = [None for _ in range(self.numQuestions)]
        per_question_predicted = [None for _ in range(self.numQuestions)]

        per_question_agent_acc = [None for _ in range(self.numQuestions)]
        per_question_option_acc = [None for _ in range(self.numQuestions)]
        # print(params)

       
        for ind in range(len(probabilities)):
            loss_option, acc_option, predicted, ground = self.computeHumanLoss(params,probabilities[ind][:4],label[ind],lossType,loss_function,modelName)
            per_question_option_loss[ind] = loss_option
            per_question_option_acc[ind] = acc_option
            per_question_ground[ind] = ground
            per_question_predicted[ind] = predicted



        return per_question_option_loss,per_question_ground,per_question_predicted

    def computeAverageLossPerTeam1(self,params, probabilities, label, lossType, loss_function, modelName):
        # print(params)
  
        # agent_prob = np.array([0.25,0.25,0.25,0.25])
        # human_prob = np.array([0.25,0.25,0.25,0.25])
        # human_prob = np.asarray(params)[:4]

        

        (per_question_option_loss,ground,predicted) = self.computeCPTLoss1(params,probabilities,label,lossType,loss_function,modelName)



        agent_loss = 0
        option_loss = 0
        option_all_loss = []
        agent_all_loss = []
        option_all_acc = []
        agent_all_acc = []
        agent_count = 0
        option_count = 0

        agent_acc = 0
        option_acc = 0
        agent_count_acc = 0
        option_count_acc = 0


        for optionLoss in per_question_option_loss:
            if (optionLoss != None):
                option_all_loss.append(optionLoss)

        ground1 = []
        predicted1 = []
        for ele in ground:
            if ele != None:
                ground1.append(ele)

        for ele in predicted:
            if ele != None:
                predicted1.append(ele)


        return option_all_loss,ground1,predicted1

if __name__ == '__main__':
    directory = "logs/"
#     cleanEventLog(directory+"event_log.csv")
#     insertInfluenceMatrixNumber(directory+"event_log-Copy.csv")
#     addMissingLogs(directory, directory+"event_log.csv")
    testSize = 15
    batchNumbers = [10,11,12,13,17,20,21,28,30,33,34,36,37,38,39,41,42,43,44,45,48,49,74,75,77,82,84,85,87,88]
    RATIONAL_PARAMS= (1,1,1)
    NUM_CPT_PARAMS = 2
    NUM_QUESTIONS = 45

    team = pd.read_csv(directory+"team.csv",sep=',',quotechar="|",names=["id","sessionId","roundId", "taskId"])

    nbLoss = list()
    centLoss = list()
    nbPTLoss = list()
    centPTLoss = list()
    rdLoss = list()

    lossOption = list()
    lossAgent = list()


    nbAlpha = list()
    nbGamma = list()
    centAlpha = list()
    centGamma = list()
    
    rdOptionLoss = list()
    rdAgentLoss = list()
    nbOptionLoss = list()
    nbAgentLoss = list()
    centOptionLoss = list()
    centAgentLoss = list()
    nbPTOptionLoss = list()
    nbPTAgentLoss = list()
    centPTOptionLoss = list()
    centPTAgentLoss = list()
    total_group_accuracy = list()
    teamAccuracies = list()
    allBestHumanAccuracy = list()
    allBestMachineAccuracy = list()
    group_accuracy_per_question = list()
    group_accuracy_over_time = np.zeros(NUM_QUESTIONS)

    #lossType = "logit"
    lossType = "softmax"

    # loss_function = "binary"
    loss_function = "variational_H_qp"#used in the paper
    # loss_function = "variational_H_pq"

    option_results = []
    agent_results = []

    best_parameters = []
    prob_test = []

    all_data = []


    ind_team = [1, 2, 3, 4, 8, 10, 11, 15, 16, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 46, 47, 48, 51, 52, 53, 55, 56]

    #shuffe_ind_team = np.random.permutation(ind_team);
    shuffe_ind_team = np.asarray([28,55,31,29,48,20,3,47,56,16,25,34,8,24,52,15,22,11,10,4,35,53,30,23,46,27,2,51,1,19])
    print("shuffled index: ")
    print(shuffe_ind_team)
    theta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    tr_num_team = 10
    all_losses_NB = []
    all_losses_CENT = []
    for thea in theta:
        team_loss_NB = []
        team_loss_CENT = []
        for ind in range(tr_num_team):
            i = shuffe_ind_team[ind]
            # print("i: ")
            # print(i)
            # print("Values of team", team.iloc[i]['id'])

            #Create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)
            expectedPerformance,rewardsNB1,rewardsCENT1,probabilitiesNB1,probabilitiesCENT1,probabilitiesRANDOM = model.computePerformance()
            # print("probabilitiesNB1")
            # print(probabilitiesNB1)
            # print("probabilitiesCENT1")
            # print(probabilitiesCENT1)

            # for ind1 in range(45):
            #     if model.actionTaken[ind1] == 4:
            #         prob = probabilitiesNB1[ind1]
            #         prob1 = prob[0:4]
            #         prob1 = np.sort(prob1)
            #         diff = prob1[3]-prob1[2]

            #         RAND = random.uniform(0, 1)
            #         if diff <= 0.3 or (diff > 0.3 and RAND <= max(prob1) * thea):
            #             model.actionTaken[ind1] = 4
            #         else:
            #             model.actionTaken[ind1] = model.actionTaken1[ind1]


            # all_loss, loss, acc, ground,predicted, q_ind_not_con = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesRANDOM[model.trainingSetSize:],lossType,loss_function,"random")
            # for total_loss in all_loss:
            #     all_losses.append(total_loss)   
            # # loss = model.computeAverageLossPerTeam(expectedPerformance,probabilitiesNB1,lossType,loss_function,"random")
            # rdLoss.append(loss)
            # # Compute losses for NB and CENT
            # # print(RATIONAL_PARAMS)
            # rdAcc.append(acc)


            #start
            all_loss1, all_loss_option1, all_loss_agent1, loss1, option_loss1, agent_loss1, acc1, ground1,predicted1,exploit1,new_prob1, q_ind_not_con1  = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1,thea, lossType,loss_function,"nb")
            all_loss2, all_loss_option2, all_loss_agent2, loss2, option_loss2, agent_loss2, acc2, ground2,predicted2,exploit2,new_prob2, q_ind_not_con2  = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1,thea, lossType,loss_function,"cent")
            if all_loss1:
                for loss in all_loss1:
                    if loss != -1:
                        team_loss_NB.append(loss)

            if all_loss2:
                for loss in all_loss2:
                    if loss != -1:
                        team_loss_CENT.append(loss)


        all_losses_NB.append(np.mean(team_loss_NB))
        all_losses_CENT.append(np.mean(team_loss_CENT))
    min_index_NB = np.argmin(np.asarray(all_losses_NB))
    thea_NB = theta[min_index_NB]
    min_index_CENT = np.argmin(np.asarray(all_losses_CENT))
    thea_CENT = theta[min_index_CENT]
    print("the best theta for NB:")
    print(thea_NB)
    print("the best theta for CENT:")
    print(thea_CENT)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')

    ######======training the best beta for the decision task of the CENT model======######
    betaa = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    all_team_losses = []
    for bee in betaa:
        team_loss = []
        for ind in range(tr_num_team):
            i = shuffe_ind_team[ind]
            # print("i: ")
            # print(i)
            # print("Values of team", team.iloc[i]['id'])

            #Create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)
            _,_,_,_,probabilitiesCENT1,probabilitiesRANDOM = model.computePerformance()
            _, _, _, _, _, _, _, ground,predicted,exploit,_, _  = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1,thea_CENT, lossType,loss_function,"cent")
            model1 = HumanDecisionModels(team.iloc[i]['id'], directory)
            expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT2, group_accuracy_per_question, best_human_accuracy = model1.calculateRewards_CENT(bee,exploit)
            
            actionTaken = model1.actionTaken
            actionTaken1 = model1.actionTaken1
            probCENT = []
            label = []
            for ind1 in range(45):
                if actionTaken[ind1] == 4:
                    probCENT.append(probabilitiesCENT2[ind1][:4])
                    # if exploit1[ind1] == 0:
                    label.append(actionTaken1[ind1])
                    # else:
                    #     label.append(actionTaken[ind1])

            all_loss,ground,predicted  = model1.computeAverageLossPerTeam1(RATIONAL_PARAMS,probCENT,label,lossType,loss_function,"cent")
            if all_loss:
                for loss in all_loss:
                    if loss != -1:
                        team_loss.append(loss)

        if team_loss:
            all_team_losses.append(np.mean(team_loss))
    min_index = np.argmin(np.asarray(all_team_losses))
    bee_b = betaa[min_index]
    print("the best beta:")
    print(bee_b)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')


    ######======test procedure of decision tasks 1 and 2 of model NB======######
    NB_all_team_losses_d1 = []
    NB_all_team_losses_d2 = []
    NB_ground_all_d1 = []
    NB_predict_all_d1 = []
    NB_ground_all_d2 = []
    NB_predict_all_d2 = []
    NB_ground_all_d1_explore = []
    NB_ground_all_d1_exploit = []
    NB_predict_all_d1_explore = []
    NB_predict_all_d1_exploit = []
    NB_ground_all_d2_explore = []
    NB_ground_all_d2_exploit = []
    NB_predict_all_d2_explore = []
    NB_predict_all_d2_exploit = []

    PT_NB_all_team_losses = []
    PT_NB_ground_all = []
    PT_NB_predict_all = []
    PT_NB_ground_all_explore = []
    PT_NB_predict_all_explore = []
    PT_NB_ground_all_exploit = []
    PT_NB_predict_all_exploit = []

    for ind in range(tr_num_team,30):
        nb_team_loss_d1 = []
        nb_team_loss_d2 = []
        pt_nb_team_loss = []
        i = shuffe_ind_team[ind]
        # print("i: ")
        # print(i)
        # print("Values of team", team.iloc[i]['id'])

        #Decision Task 1
        model1 = HumanDecisionModels(team.iloc[i]['id'], directory)
        expectedPerformance,rewardsNB1,rewardsCENT1,probabilitiesNB1,probabilitiesCENT1,probabilitiesRANDOM = model1.computePerformance()

        all_loss, all_loss_option, all_loss_agent, loss, option_loss, agent_loss, acc, ground_d1,predicted_d1, exploit1,new_prob, q_ind_not_con  = model1.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1[model1.trainingSetSize:],thea_NB,lossType,loss_function,"nb")
        # print("all_loss")
        # print(all_loss)

        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    nb_team_loss_d1.append(loss)
        
        if nb_team_loss_d1:
            NB_all_team_losses_d1.append(np.mean(nb_team_loss_d1))

        for ii in range(model1.trainingSetSize,45):
            if ground_d1[ii] != None:
                NB_ground_all_d1.append(ground_d1[ii])
                NB_predict_all_d1.append(predicted_d1[ii])
                if exploit1[ii] == 0:
                    NB_ground_all_d1_explore.append(ground_d1[ii])
                    NB_predict_all_d1_explore.append(predicted_d1[ii])
                else:
                    NB_ground_all_d1_exploit.append(ground_d1[ii])
                    NB_predict_all_d1_exploit.append(predicted_d1[ii])



        b_param = []
        # Train alpha,gammma losses for NB-PT
        Alpha, Beta, Lambda, GammaGain, GammaLoss = model1.chooseCPTParameters(probabilitiesNB1[:model1.trainingSetSize],thea_NB,lossType,loss_function,"nb-pt")
        # print("PT-NB",Alpha, Beta, Lambda, GammaGain, GammaLoss)
        b_param.append(round(Alpha,1))
        b_param.append(round(Beta,1))
        b_param.append(round(Lambda,1))
        b_param.append(round(GammaGain,1))
        b_param.append(round(GammaLoss,1))
        # print("PT-NB:")
        # print(b_param)
        all_loss, all_loss_option, all_loss_agent, loss, option_loss, agent_loss, acc, ground_d1,predicted_d1, exploit2,new_prob, q_ind_not_con  = model1.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesNB1[model1.trainingSetSize:],thea_NB,lossType,loss_function,"nb-pt")
        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    pt_nb_team_loss.append(loss)
        
        if pt_nb_team_loss:
            PT_NB_all_team_losses.append(np.mean(pt_nb_team_loss))

        for ii in range(model1.trainingSetSize,45):
            if ground_d1[ii] != None:
                PT_NB_ground_all.append(ground_d1[ii])
                PT_NB_predict_all.append(predicted_d1[ii])
                if exploit2[ii] == 0:
                    PT_NB_ground_all_explore.append(ground_d1[ii])
                    PT_NB_predict_all_explore.append(predicted_d1[ii])
                else:
                    PT_NB_ground_all_exploit.append(ground_d1[ii])
                    PT_NB_predict_all_exploit.append(predicted_d1[ii])



        ######======Decision Task 2======######
        expectedPerformance,rewardsNB1,rewardsCENT1,probabilitiesNB1,probabilitiesCENT1 = model1.calculateRewards_NB(exploit1)
        
        actionTaken = model1.actionTaken
        actionTaken1 = model1.actionTaken1
        probNB1 = []
        # probNB1_H = []
        # probNB1_A = []
        label = []
        label1 = []
        for ind1 in range(model1.trainingSetSize,45):
            if actionTaken[ind1] == 4:
                probNB1.append(probabilitiesNB1[ind1][:4])
                # probNB1_H.append(probabilitiesNB1_H[ind1][:4])
                # probNB1_A.append(probabilitiesNB1_A[ind1][4:])
                # if exploit1[ind1] == 0:
                label.append(actionTaken1[ind1])
                label1.append(exploit1[ind1])
                # else:
                #     label.append(actionTaken[ind1])

        all_loss,ground_d2,predicted_d2  = model.computeAverageLossPerTeam1(RATIONAL_PARAMS,probNB1,label, lossType,loss_function,"nb")
        
        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    nb_team_loss_d2.append(loss)

        if nb_team_loss_d2:
            NB_all_team_losses_d2.append(np.mean(nb_team_loss_d2))

        for ii in range(len(ground_d2)):
            if ground_d2[ii] != None:
                NB_ground_all_d2.append(ground_d2[ii])
                NB_predict_all_d2.append(predicted_d2[ii])
                if label1[ii] == 0:
                    NB_ground_all_d2_explore.append(ground_d2[ii])
                    NB_predict_all_d2_explore.append(predicted_d2[ii])
                else:
                    NB_ground_all_d2_exploit.append(ground_d2[ii])
                    NB_predict_all_d2_exploit.append(predicted_d2[ii])


    print("NB ",np.mean(NB_all_team_losses_d1),np.std(NB_all_team_losses_d1))
    print("accuracy_score:")
    print(accuracy_score(NB_ground_all_d1,NB_predict_all_d1))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(NB_ground_all_d1,NB_predict_all_d1))
    print("f1_score macro:")
    print(f1_score(NB_ground_all_d1,NB_predict_all_d1,average='macro'))
    print("f1_score micro:")
    print(f1_score(NB_ground_all_d1,NB_predict_all_d1,average='micro'))
    print("precision_score macro:")
    print(precision_score(NB_ground_all_d1,NB_predict_all_d1,average='macro'))
    print("precision_score micro:")
    print(precision_score(NB_ground_all_d1,NB_predict_all_d1,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(NB_ground_all_d1,NB_predict_all_d1,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(NB_ground_all_d1,NB_predict_all_d1,average='micro'))
    print("recall_score macro:")
    print(recall_score(NB_ground_all_d1,NB_predict_all_d1,average='macro'))
    print("recall_score micro:")
    print(recall_score(NB_ground_all_d1,NB_predict_all_d1,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(NB_ground_all_d1,NB_predict_all_d1,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(NB_ground_all_d1,NB_predict_all_d1,average='micro'))

    print("confusion matrix D1:")
    print(confusion_matrix(NB_ground_all_d1, NB_predict_all_d1))
    print(NB_ground_all_d1)
    print(NB_predict_all_d1)
    print("confusion matrix D1 explore:")
    print(confusion_matrix(NB_ground_all_d1_explore, NB_predict_all_d1_explore))
    print(NB_ground_all_d1_explore)
    print(NB_predict_all_d1_explore)
    print("confusion matrix D1 exploit:")
    print(confusion_matrix(NB_ground_all_d1_exploit, NB_predict_all_d1_exploit))
    print(NB_ground_all_d1_exploit)
    print(NB_predict_all_d1_exploit)

    print("NB ",np.mean(NB_all_team_losses_d2),np.std(NB_all_team_losses_d2))
    print("accuracy_score:")
    print(accuracy_score(NB_ground_all_d2,NB_predict_all_d2))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(NB_ground_all_d2,NB_predict_all_d2))
    print("f1_score macro:")
    print(f1_score(NB_ground_all_d2,NB_predict_all_d2,average='macro'))
    print("f1_score micro:")
    print(f1_score(NB_ground_all_d2,NB_predict_all_d2,average='micro'))
    print("precision_score macro:")
    print(precision_score(NB_ground_all_d2,NB_predict_all_d2,average='macro'))
    print("precision_score micro:")
    print(precision_score(NB_ground_all_d2,NB_predict_all_d2,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(NB_ground_all_d2,NB_predict_all_d2,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(NB_ground_all_d2,NB_predict_all_d2,average='micro'))
    print("recall_score macro:")
    print(recall_score(NB_ground_all_d2,NB_predict_all_d2,average='macro'))
    print("recall_score micro:")
    print(recall_score(NB_ground_all_d2,NB_predict_all_d2,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(NB_ground_all_d2,NB_predict_all_d2,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(NB_ground_all_d2,NB_predict_all_d2,average='micro'))

    print("confusion matrix D2:")
    print(confusion_matrix(NB_ground_all_d2, NB_predict_all_d2))
    print(NB_ground_all_d2)
    print(NB_predict_all_d2)
    print("confusion matrix D2 explore:")
    print(confusion_matrix(NB_ground_all_d2_explore, NB_predict_all_d2_explore))
    print(NB_ground_all_d2_explore)
    print(NB_predict_all_d2_explore)
    print("confusion matrix D2 exploit:")
    print(confusion_matrix(NB_ground_all_d2_exploit, NB_predict_all_d2_exploit))
    print(NB_ground_all_d2_exploit)
    print(NB_predict_all_d2_exploit)

    print("PT-NB ",np.mean(PT_NB_all_team_losses),np.std(PT_NB_all_team_losses))
    print("accuracy_score:")
    print(accuracy_score(PT_NB_ground_all,PT_NB_predict_all))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(PT_NB_ground_all,PT_NB_predict_all))
    print("f1_score macro:")
    print(f1_score(PT_NB_ground_all,PT_NB_predict_all,average='macro'))
    print("f1_score micro:")
    print(f1_score(PT_NB_ground_all,PT_NB_predict_all,average='micro'))
    print("precision_score macro:")
    print(precision_score(PT_NB_ground_all,PT_NB_predict_all,average='macro'))
    print("precision_score micro:")
    print(precision_score(PT_NB_ground_all,PT_NB_predict_all,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(PT_NB_ground_all,PT_NB_predict_all,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(PT_NB_ground_all,PT_NB_predict_all,average='micro'))
    print("recall_score macro:")
    print(recall_score(PT_NB_ground_all,PT_NB_predict_all,average='macro'))
    print("recall_score micro:")
    print(recall_score(PT_NB_ground_all,PT_NB_predict_all,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(PT_NB_ground_all,PT_NB_predict_all,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(PT_NB_ground_all,PT_NB_predict_all,average='micro'))

    print("confusion matrix D1:")
    print(confusion_matrix(PT_NB_ground_all, PT_NB_predict_all))
    print(PT_NB_ground_all)
    print(PT_NB_predict_all)
    print("confusion matrix D1 explore:")
    print(confusion_matrix(PT_NB_ground_all_explore, PT_NB_predict_all_explore))
    print(PT_NB_ground_all_explore)
    print(PT_NB_predict_all_explore)
    print("confusion matrix D1 exploit:")
    print(confusion_matrix(PT_NB_ground_all_exploit, PT_NB_predict_all_exploit))
    print(PT_NB_ground_all_exploit)
    print(PT_NB_predict_all_exploit)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')



    ######======test procedure of decision tasks 1 and 2 of model CENT======######
    CENT_all_team_losses_d1 = []
    CENT_all_team_losses_d2 = []
    CENT_ground_all_d1 = []
    CENT_predict_all_d1 = []
    CENT_ground_all_d1_explore = []
    CENT_predict_all_d1_explore = []
    CENT_ground_all_d1_exploit = []
    CENT_predict_all_d1_exploit = []
    CENT_ground_all_d2 = []
    CENT_predict_all_d2 = []
    CENT_ground_all_d2_explore = []
    CENT_predict_all_d2_explore = []
    CENT_ground_all_d2_exploit = []
    CENT_predict_all_d2_exploit = []

    PT_CENT_all_team_losses = []
    PT_CENT_ground_all = []
    PT_CENT_predict_all = []
    PT_CENT_ground_all_explore = []
    PT_CENT_predict_all_explore = []
    PT_CENT_ground_all_exploit = []
    PT_CENT_predict_all_exploit = []

    for ind in range(tr_num_team,30):
        team_loss_d1 = []
        team_loss_d2 = []
        cent_team_loss_d1 = []
        cent_team_loss_d2 = []
        pt_cent_team_loss = []
        i = shuffe_ind_team[ind]
        # print("i: ")
        # print(i)
        # print("Values of team", team.iloc[i]['id'])

        #Decision Task 1
        model2 = HumanDecisionModels(team.iloc[i]['id'], directory)
        expectedPerformance,rewardsNB1,rewardsCENT1,probabilitiesNB1,probabilitiesCENT1,probabilitiesRANDOM = model2.computePerformance()

        all_loss, all_loss_option, all_loss_agent, loss, option_loss, agent_loss, acc, ground_d1,predicted_d1, exploit1,new_prob, q_ind_not_con  = model2.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1[model2.trainingSetSize:],thea_CENT,lossType,loss_function,"cent")
        # print("all_loss")
        # print(all_loss)
        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    cent_team_loss_d1.append(loss)
        
        if cent_team_loss_d1:
            CENT_all_team_losses_d1.append(np.mean(cent_team_loss_d1))

        for ii in range(model2.trainingSetSize,45):
            if ground_d1[ii] != None:
                CENT_ground_all_d1.append(ground_d1[ii])
                CENT_predict_all_d1.append(predicted_d1[ii])
                if exploit1[ii] == 0:
                    CENT_ground_all_d1_explore.append(ground_d1[ii])
                    CENT_predict_all_d1_explore.append(predicted_d1[ii])
                else:
                    CENT_ground_all_d1_exploit.append(ground_d1[ii])
                    CENT_predict_all_d1_exploit.append(predicted_d1[ii])


        b_param = []
        # Train alpha,gammma losses for CENT-PT
        Alpha, Beta, Lambda, GammaGain, GammaLoss = model2.chooseCPTParameters(probabilitiesCENT1[:model2.trainingSetSize],thea_CENT,lossType,loss_function,"cent-pt")
        # print("CENT-PT",Alpha, Beta, Lambda, GammaGain, GammaLoss)
        b_param.append(round(Alpha,1))
        b_param.append(round(Beta,1))
        b_param.append(round(Lambda,1))
        b_param.append(round(GammaGain,1))
        b_param.append(round(GammaLoss,1))
        print("PT-CENT:")
        print(b_param)
        all_loss, all_loss_option, all_loss_agent, loss, option_loss, agent_loss, acc, ground_d1,predicted_d1, exploit2,new_prob, q_ind_not_con  = model2.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesCENT1[model2.trainingSetSize:],thea_CENT,lossType,loss_function,"cent-pt")
        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    pt_cent_team_loss.append(loss)
        
        if pt_cent_team_loss:
            PT_CENT_all_team_losses.append(np.mean(pt_cent_team_loss))

        for ii in range(model2.trainingSetSize,45):
            if ground_d1[ii] != None:
                PT_CENT_ground_all.append(ground_d1[ii])
                PT_CENT_predict_all.append(predicted_d1[ii])
                if exploit2[ii] == 0:
                    PT_CENT_ground_all_explore.append(ground_d1[ii])
                    PT_CENT_predict_all_explore.append(predicted_d1[ii])
                else:
                    PT_CENT_ground_all_exploit.append(ground_d1[ii])
                    PT_CENT_predict_all_exploit.append(predicted_d1[ii])

        ######======Decision Task 2======######
        model3 = HumanDecisionModels(team.iloc[i]['id'], directory)
        expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1, group_accuracy_per_question, best_human_accuracy = model3.calculateRewards_CENT(bee_b,exploit1)
            
        actionTaken = model3.actionTaken
        actionTaken1 = model3.actionTaken1
        probCENT = []
        label = []
        label1 = []
        for ind1 in range(model3.trainingSetSize,45):
            if actionTaken[ind1] == 4:
                probCENT.append(probabilitiesCENT1[ind1][:4])
                # if exploit1[ind1] == 0:
                label.append(actionTaken1[ind1])
                label1.append(exploit1[ind1])
                # else:
                #     label.append(actionTaken[ind1])

        all_loss,ground_d2,predicted_d2,  = model3.computeAverageLossPerTeam1(RATIONAL_PARAMS,probCENT,label, lossType,loss_function,"cent")
        if all_loss:
            for loss in all_loss:
                if loss != -1:
                    cent_team_loss_d2.append(loss)

        if cent_team_loss_d2:
            CENT_all_team_losses_d2.append(np.mean(cent_team_loss_d2))

        for ii in range(len(ground_d2)):
            if ground_d2[ii] != None:
                CENT_ground_all_d2.append(ground_d2[ii])
                CENT_predict_all_d2.append(predicted_d2[ii])
                if label1[ii] == 0:
                    CENT_ground_all_d2_explore.append(ground_d2[ii])
                    CENT_predict_all_d2_explore.append(predicted_d2[ii])
                else:
                    CENT_ground_all_d2_exploit.append(ground_d2[ii])
                    CENT_predict_all_d2_exploit.append(predicted_d2[ii])

    print("CENT ",np.mean(CENT_all_team_losses_d1),np.std(CENT_all_team_losses_d1))
    print("accuracy_score:")
    print(accuracy_score(CENT_ground_all_d1,CENT_predict_all_d1))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(CENT_ground_all_d1,CENT_predict_all_d1))
    print("f1_score macro:")
    print(f1_score(CENT_ground_all_d1,CENT_predict_all_d1,average='macro'))
    print("f1_score micro:")
    print(f1_score(CENT_ground_all_d1,CENT_predict_all_d1,average='micro'))
    print("precision_score macro:")
    print(precision_score(CENT_ground_all_d1,CENT_predict_all_d1,average='macro'))
    print("precision_score micro:")
    print(precision_score(CENT_ground_all_d1,CENT_predict_all_d1,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(CENT_ground_all_d1,CENT_predict_all_d1,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(CENT_ground_all_d1,CENT_predict_all_d1,average='micro'))
    print("recall_score macro:")
    print(recall_score(CENT_ground_all_d1,CENT_predict_all_d1,average='macro'))
    print("recall_score micro:")
    print(recall_score(CENT_ground_all_d1,CENT_predict_all_d1,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(CENT_ground_all_d1,CENT_predict_all_d1,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(CENT_ground_all_d1,CENT_predict_all_d1,average='micro'))

    print("confusion matrix D1:")
    print(confusion_matrix(CENT_ground_all_d1, CENT_predict_all_d1))
    print(CENT_ground_all_d1)
    print(CENT_predict_all_d1)
    print("confusion matrix D1 explore:")
    print(confusion_matrix(CENT_ground_all_d1_explore, CENT_predict_all_d1_explore))
    print(CENT_ground_all_d1_explore)
    print(CENT_predict_all_d1_explore)
    print("confusion matrix D1 exploit:")
    print(confusion_matrix(CENT_ground_all_d1_exploit, CENT_predict_all_d1_exploit))
    print(CENT_ground_all_d1_exploit)
    print(CENT_predict_all_d1_exploit)

    print("CENT ",np.mean(CENT_all_team_losses_d2),np.std(CENT_all_team_losses_d2))
    print("accuracy_score:")
    print(accuracy_score(CENT_ground_all_d2,CENT_predict_all_d2))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(CENT_ground_all_d2,CENT_predict_all_d2))
    print("f1_score macro:")
    print(f1_score(CENT_ground_all_d2,CENT_predict_all_d2,average='macro'))
    print("f1_score micro:")
    print(f1_score(CENT_ground_all_d2,CENT_predict_all_d2,average='micro'))
    print("precision_score macro:")
    print(precision_score(CENT_ground_all_d2,CENT_predict_all_d2,average='macro'))
    print("precision_score micro:")
    print(precision_score(CENT_ground_all_d2,CENT_predict_all_d2,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(CENT_ground_all_d2,CENT_predict_all_d2,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(CENT_ground_all_d2,CENT_predict_all_d2,average='micro'))
    print("recall_score macro:")
    print(recall_score(CENT_ground_all_d2,CENT_predict_all_d2,average='macro'))
    print("recall_score micro:")
    print(recall_score(CENT_ground_all_d2,CENT_predict_all_d2,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(CENT_ground_all_d2,CENT_predict_all_d2,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(CENT_ground_all_d2,CENT_predict_all_d2,average='micro'))

    print("confusion matrix D2:")
    print(confusion_matrix(CENT_ground_all_d2, CENT_predict_all_d2))
    print(CENT_ground_all_d2)
    print(CENT_predict_all_d2)
    print("confusion matrix D2 explore:")
    print(confusion_matrix(CENT_ground_all_d2_explore, CENT_predict_all_d2_explore))
    print(CENT_ground_all_d2_explore)
    print(CENT_predict_all_d2_explore)
    print("confusion matrix D2 exploit:")
    print(confusion_matrix(CENT_ground_all_d2_exploit, CENT_predict_all_d2_exploit))
    print(CENT_ground_all_d2_exploit)
    print(CENT_predict_all_d2_exploit)

    print("PT-CENT ",np.mean(PT_CENT_all_team_losses),np.std(PT_CENT_all_team_losses))
    print("accuracy_score:")
    print(accuracy_score(PT_CENT_ground_all,PT_CENT_predict_all))
    print("balanced_accuracy_score:")
    print(balanced_accuracy_score(PT_CENT_ground_all,PT_CENT_predict_all))
    print("f1_score macro:")
    print(f1_score(PT_CENT_ground_all,PT_CENT_predict_all,average='macro'))
    print("f1_score micro:")
    print(f1_score(PT_CENT_ground_all,PT_CENT_predict_all,average='micro'))
    print("precision_score macro:")
    print(precision_score(PT_CENT_ground_all,PT_CENT_predict_all,average='macro'))
    print("precision_score micro:")
    print(precision_score(PT_CENT_ground_all,PT_CENT_predict_all,average='micro'))
    # print("average_precision_score macro:")
    # print(average_precision_score(PT_CENT_ground_all,PT_CENT_predict_all,average='macro'))
    # print("average_precision_score micro:")
    # print(average_precision_score(PT_CENT_ground_all,PT_CENT_predict_all,average='micro'))
    print("recall_score macro:")
    print(recall_score(PT_CENT_ground_all,PT_CENT_predict_all,average='macro'))
    print("recall_score micro:")
    print(recall_score(PT_CENT_ground_all,PT_CENT_predict_all,average='micro'))
    # print("roc_auc_score macro:")
    # print(roc_auc_score(PT_CENT_ground_all,PT_CENT_predict_all,average='macro'))
    # print("roc_auc_score micro:")
    # print(roc_auc_score(PT_CENT_ground_all,PT_CENT_predict_all,average='micro'))

    print("confusion matrix D1:")
    print(confusion_matrix(PT_CENT_ground_all, PT_CENT_predict_all))
    print(PT_CENT_ground_all)
    print(PT_CENT_predict_all)
    print("confusion matrix D1 explore:")
    print(confusion_matrix(PT_CENT_ground_all_explore, PT_CENT_predict_all_explore))
    print(PT_CENT_ground_all_explore)
    print(PT_CENT_predict_all_explore)
    print("confusion matrix D1 exploit:")
    print(confusion_matrix(PT_CENT_ground_all_exploit, PT_CENT_predict_all_exploit))
    print(PT_CENT_ground_all_exploit)
    print(PT_CENT_predict_all_exploit)

  
    #Wilcoxson signed-rank test
    print(stats.mannwhitneyu(np.asarray(NB_all_team_losses_d1),np.asarray(CENT_all_team_losses_d1)))
    print(stats.mannwhitneyu(np.asarray(NB_all_team_losses_d1),np.asarray(PT_NB_all_team_losses)))
    print(stats.mannwhitneyu(np.asarray(NB_all_team_losses_d1),np.asarray(PT_CENT_all_team_losses)))
    print(stats.mannwhitneyu(np.asarray(CENT_all_team_losses_d1),np.asarray(PT_NB_all_team_losses)))
    print(stats.mannwhitneyu(np.asarray(CENT_all_team_losses_d1),np.asarray(PT_CENT_all_team_losses)))
    print(stats.mannwhitneyu(np.asarray(PT_NB_all_team_losses),np.asarray(PT_CENT_all_team_losses)))
    print('here\n')
    print(stats.mannwhitneyu(np.asarray(NB_all_team_losses_d2),np.asarray(CENT_all_team_losses_d2)))

    print("NB_all_team_losses_d1")
    print(NB_all_team_losses_d1)
    print("NB_all_team_losses_d2")
    print(NB_all_team_losses_d2)
    print("CENT_all_team_losses_d1")
    print(CENT_all_team_losses_d1)
    print("CENT_all_team_losses_d2")
    print(CENT_all_team_losses_d2)
    print("PT_NB_all_team_losses")
    print(PT_NB_all_team_losses)
    print("PT_CENT_all_team_losses")
    print(PT_CENT_all_team_losses)
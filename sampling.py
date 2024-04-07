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
import AnalysisFunctions as plf
from scipy.stats import wilcoxon
from statistics import mean
from scipy.stats import pearsonr
from cpt_valuation import evaluateProspectVals
from sklearn.metrics import precision_recall_fscore_support
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle


def computeLoss(soft_prob,chosen_action,loss_function):
        
    predicted = np.argmax(soft_prob)
    ground = chosen_action

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
        
        if i == j:
            for k in range(8):
                arg = soft_prob[k]
                loss += -arg * math.log(arg) 
        #sampling
        else:

            variation_q = np.zeros(8)
            variation_q[j] = random.uniform(0, 1)
            index1 = []
            index2 = []
            index1.append(j)
            for k in range(8):
                index2.append(int(k))
            index3 = list(set(index2) - set(index1))
            left = 1 -  variation_q[j]
            for k in range(6):
                variation_q[index3[k]] = random.uniform(0, left)
                left -= variation_q[index3[k]]
            variation_q[index3[6]] = left

            m_ind = np.argmax(variation_q)

            if j != m_ind:
                tmp = variation_q[j]
                variation_q[j] = variation_q[m_ind]
                variation_q[m_ind] = tmp
            
            # print(sum(variation_q))
            # print(np.argmax(variation_q))
            for k in range(8):
                loss += -soft_prob[k] * math.log(variation_q[k])

    else:

        #sampling
        i = np.argmax(soft_prob)
        j = chosen_action

        if i == j:
            for k in range(8):
                arg = soft_prob[k]
                loss += -arg * math.log(arg) 
        #sampling
        else:

            variation_q =  np.zeros(8)
            variation_q[j] = random.uniform(0, 1)
            index1 = []
            index2 = []
            index1.append(j)
            for k in range(8):
                index2.append(int(k))
            index3 = list(set(index2) - set(index1))
            left = 1 -  variation_q[j]
            for k in range(6):
                variation_q[index3[k]] = random.uniform(0, left)
                left -= variation_q[index3[k]]
            variation_q[index3[6]] = left

            m_ind = np.argmax(variation_q)

            if j != m_ind:
                tmp = variation_q[j]
                variation_q[j] = variation_q[m_ind]
                variation_q[m_ind] = tmp

            # print(sum(variation_q))
            # print(np.argmax(variation_q))

            for k in range(8):
                loss += -variation_q[k] * math.log(soft_prob[k])


    return loss

def histPlot(data):
    mu = np.mean(data) # mean of distribution
    sigma = np.std(data) # standard deviation of distribution
    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b-')


    plt.xlabel('L2')
    plt.ylabel('PDF')
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()


def histogramPlot(data):
    mu = np.mean(data[0]) # mean of distribution
    sigma = np.std(data[0]) # standard deviation of distribution
    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(data[0], bins=np.arange(0.05,20,0.05), facecolor='blue', alpha=0.5, label = '$L^{(1)}$')
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b-')

    mu = np.mean(data[1]) # mean of distribution
    sigma = np.std(data[1]) # standard deviation of distribution
    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(data[1], bins=np.arange(0.05,20,0.05), facecolor='red', alpha=0.5, label = '$L^{(2)}$')
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r-')


    plt.xlabel('Loss')
    plt.ylabel('PDF')
    # plt.title('Histogram - '+modelName)
    plt.legend(loc='upper right')
    # # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    # plt.savefig('Histogram-'+modelName+'-'+lossType)
    plt.show()
    plt.clf()


if __name__ == '__main__':

    loss_function1 = "variational_H_qp"
    loss_function2 = "variational_H_pq"
    runtime = 1000
    loss1 = []
    loss2 = []
    # soft_prob = [0.09980511, 0.09980511, 0.18431119, 0.12109884, 0.12249696, 0.12970566, 0.12063496, 0.12214217]
    # chosen_action = 3
    with open('./probability.data', 'rb') as filehandle:
        # read the data as binary data stream
        all_soft_prob = pickle.load(filehandle)

    min_l1 = []
    min_l2 = []


    for ele in all_soft_prob:
        chosen_action = ele[-1]
        soft_prob = ele[:-1]
        l11 = []
        l21 = []
        for i in range(runtime):
            l1 = computeLoss(soft_prob,chosen_action,loss_function1)
            l11.append(l1)
            loss1.append(l1)
            l2 = computeLoss(soft_prob,chosen_action,loss_function2)
            l21.append(l2)
            loss2.append(l2)
        min_l1.append(np.amin(l11))
        min_l2.append(np.amin(l21))

    print("mean L1")
    print(np.mean(loss1))
    print("mean L2")
    print(np.mean(loss2))
    print("min L1")
    print(np.mean(min_l1))
    print("min L2")
    print(np.mean(min_l2))
    data = []
    data.append(loss1)
    data.append(loss2)
    histogramPlot(data)



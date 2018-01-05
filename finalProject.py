import copy
import os
import pyshark
from Vectorization import changetovector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Packet import Packet

def log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('KDD2009')
    return logger

logger = log()


def append_burst(target, element):
    list = copy.copy(element)
    target.append(list)
    element.clear()


def append_packet(up_list, down_list, bi_list, packet):
    bi_list.append(packet)
    if packet.len > 0:
        up_list.append(packet)
    else:
        down_list.append(packet)


def get_burst(cap):
    sum = 0
    time = 0
    list_up_packet = []
    list_down_packet = []
    list_bi_packet = []
    list = []
    list_burst = []
    idx = 0
    for _ in cap:
        # parse the packet
        sum += int(_.ip.len)
        length = float(_.captured_length)
        dst = str(_.layers[1].dst)
        src = str(_.layers[1].src)
        # the first packet, get the source ip and destination ip
        if idx == 0:
            time = float(_.sniff_timestamp)
            Packet.source = src
            Packet.destination = dst
            packet = Packet(src, dst, length)
            list_up_packet.append(packet)
            list_bi_packet.append(packet)
        else:
            packet = Packet(src, dst, length)
            tmp_time = float(_.sniff_timestamp)

            # print(tmp_time)
            # start a new burst
            if tmp_time - time > 0.1:
                # copy and append the previous burst
                append_burst(list,list_up_packet)
                append_burst(list, list_down_packet)
                append_burst(list, list_bi_packet)
                list_burst.append(list)
                list = []
                append_packet(list_up_packet, list_down_packet, list_bi_packet, packet)
            #     append the existing burst
            else:
                append_packet(list_up_packet, list_down_packet, list_bi_packet, packet)
            time = tmp_time
        idx += 1
    append_burst(list, list_up_packet)
    append_burst(list, list_down_packet)
    append_burst(list, list_bi_packet)
    list_burst.append(list)
    return list_burst


def get_data(list_burst):
    for burst in list_burst:
        for list in burst:
            for i in range(len(list)):
                list[i] = list[i].len


def load_data(path):
    list_res = []
    listburst=[]
    for file in os.listdir(path):
        try:
            cap = pyshark.FileCapture(path + file)
            listburst = get_burst(cap)
            get_data(listburst)
            list_res.extend(listburst)
        except:
            print("not a file")

    return list_res

def get_feature(data):
    feature = []
    for tmp_res in data:
        tmp = changetovector(tmp_res)
        feature.append(tmp)
    return feature

def add_label(data,label):
    tmp = 0
    res = []
    if label == 'social':
        tmp = 0
    elif label == 'communication':
        tmp = 1
    else:
        tmp = 2

    for i in range(len(data)):
        res.append(tmp)
    return res

def scalar(data):
    #
    datapd = pd.DataFrame(data)
    data_norm = (datapd-datapd.mean())/datapd.std()
    return data_norm

def main():
    # try:
    #     dataset = pd.read_csv('foo.csv').transpose
    # except:

    list_social = load_data('social/')
    print("social load finish")
    list_finance = load_data('finance/')
    print("finance load finish")
    list_communication = load_data('communication/')
    print("communication load finish")

    social_data = get_feature(list_social)
    social_data_norm = scalar(social_data)

    finance_data = get_feature(list_finance)
    finance_data_norm = scalar(finance_data)

    communication_data = get_feature(list_communication)
    communication_data_norm = scalar(communication_data)

    label_social = add_label(social_data_norm,'social')
    label_finance = add_label(finance_data_norm, 'finance')
    label_communication = add_label(communication_data_norm, 'communication')


    # concatenate data
    labels = []
    # social_data_norm.append(finance_data_norm)
    # social_data_norm.append(communication_data_norm)

    labels.extend(label_social)
    labels.extend(label_finance)
    labels.extend(label_communication)

    frames = [social_data_norm, finance_data_norm, communication_data_norm]
    dataset = pd.concat(frames, ignore_index=True)
    # dataset.to_csv("foo.csv")
    X = dataset
    labels = pd.DataFrame(labels)
    # labels.to_csv("label.csv")
    Y = labels

    ############################## Split Dataset ##############################################
    logger.info('Start to split data set into train set, validation set and test set randomly')

    # Split X,Y to train and test
    train_proportion = 0.95

    def train_validate_test_split(X, Y, train_percent):
        totLen = len(X)
        train_end = int(totLen * train_percent)
        trainX = X[:train_end]
        trainY = Y[:train_end]
        testX = X[train_end:]
        testY = Y[train_end:]
        return trainX, trainY, testX, testY

    trainX, trainY, testX, testY = train_validate_test_split(X, Y, train_proportion)

    ############################## Build Model #################################################
    # Train and test your model using (trainX, trainY), (testX, testY)
    def benchmark(clf, trainX, trainY, testX, testY, logger):
        clf_descr = str(clf).split('(')[0]
        logger.info('Start to fit %s' % clf_descr)
        # Train the clf, and record the training time
        t0 = time()
        clf.fit(trainX, trainY)
        train_time = time() - t0
        print('Training time: %0.3fs' % train_time)

        # Fit the clf to the test dataset, and record the testing time
        t0 = time()
        predict = clf.predict(testX)
        test_time = time() - t0
        print('Testing time: %0.3fs' % test_time)

        score = float(accuracy_score(testY, predict, normalize=False)) / len(testY)
        print('Accuracy of {0}: {1:.2%}'.format(clf_descr, score))

        logger.info('Finished fitting %s' % clf_descr + '\n')
        return clf_descr, score, train_time, test_time

    def drawModelComparison(results):
        indices = np.arange(len(results))
        results = [[result[i] for result in results] for i in range(4)]
        clf, score, train_time, test_time = results

        train_time = np.array(train_time) / np.max(train_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Model Comparison for bigdata")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, train_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf):
            plt.text(-.3, i, c)

        plt.savefig('Model_Comparison bigdata.png')

    # Gaussian Naive Bayes
    gnb = GaussianNB()

    # Random Forest Classifier
    rfc = RandomForestClassifier(max_depth=15, n_estimators=10)

    # Adaboosting classifier
    AdaDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=10, learning_rate=0.5)

    # Bagging classifier
    bagging_lr = BaggingClassifier(GaussianNB())

    # Result
    results = []
    for clf, name in ((gnb, 'Gaussian Naive Bayes'),
                      (rfc, 'Random Forest'),
                      (AdaDT, 'Adaboosting classifier'),
                      # (svc, 'SVM  classifier'),
                      (bagging_lr, 'Bagging')):
        results.append(benchmark(clf, trainX, trainY, testX, testY, logger))

    drawModelComparison(results)

if __name__=='__main__': main()
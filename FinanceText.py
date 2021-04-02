#  - * -  coding:utf-8 - * -
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
import numpy as np
import jieba
import networkx as nx
import matplotlib.pyplot as plt


class FinanceText:
    def __init__(self, sourcePath, targetPath, preference='投资'):
        """
        本类提供对金融文本抽取概要的功能

        :param sourcePath:原始金融文本路径
        :param targetPath:提取后金融文本概要保存路径
        :param preference:用户偏好选择
        """
        self.sourcePath = sourcePath
        self.targetPath = targetPath
        self.preference = preference
        print('当前偏好为：' + self.preference)
        self.text = self.getText()
        self.stopWords = self.getStopWords()
        self.dictionary, self.train = self.get_dict()

    def getText(self):
        """
        获取金融文本

        :return:text:金融文本
        """
        with open(self.sourcePath, mode='r', encoding='utf8') as file:
            text = file.readlines()

        return text

    def textWrite(self, text):
        """
        将传入的金融文本写入Finance.txt
        """
        with open(self.targetPath, mode='w', encoding='utf8') as file:
            file.write(text)

    def getStopWords(self):
        """
        获取stopwords

        :return:stopwords:返回stopwords
        """
        stopWords = set()

        with open('stopwords.txt', encoding='utf8', mode='r') as file:
            for line in file:
                stopWords.add(line.rstrip())

        return stopWords

    def get_dict(self):
        train = []
        for line in self.text:
            line = list(jieba.cut(line))
            train.append([w for w in line if w not in self.stopWords])
        dictionary = Dictionary(train)
        return dictionary, train

    def train_model(self):
        corpus = [self.dictionary.doc2bow(text) for text in self.train]
        lda = LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=7)
        # 模型的保存/ 加载
        lda.save('test_lda.model')

    def lda_sim(self, s1):
        lda = models.ldamodel.LdaModel.load('test_lda.model')
        test_doc = list(jieba.cut(s1))
        doc_bow = self.dictionary.doc2bow(test_doc)
        doc_lda = lda[doc_bow]  # 文档1的主题分布
        list_doc1 = [i[1] for i in doc_lda]
        test_doc2 = list(jieba.cut(self.preference))
        doc_bow2 = self.dictionary.doc2bow(test_doc2)
        doc_lda2 = lda[doc_bow2]  # 文档2的主题分布
        list_doc2 = [i[1] for i in doc_lda2]
        try:
            sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
        except ValueError:
            sim = 0
        # 文档相似度，越大越相近
        return sim

    def hanlp_split(self, text):
        """
        分隔hanlp导出的信息

        :param text: hanlp导出的信息
        :return:返回[代号, 文本, 词性, 关系指向, 关系]的数据结构与(代号, 代号)的关系组
        """
        textSplit = text.split('\n')
        textSplit.pop(-1)
        textProcessed = []
        relationGraph = []
        for t in textSplit:
            tmp = t.split('\t')
            # if tmp[1] not in self.stopWords:
            #     wordArray = [tmp[0], tmp[1], tmp[3], tmp[6], tmp[7]]
            #     relation = (tmp[0], tmp[6])
            #     textProcessed.append(wordArray)
            #     relationGraph.append(relation)
            wordArray = [tmp[0], tmp[1], tmp[3], tmp[6], tmp[7]]
            relation = (tmp[0], tmp[6])
            textProcessed.append(wordArray)
            relationGraph.append(relation)

        return textProcessed, relationGraph

    def showGraph(self, graph):
        """
        绘制语句中词汇关系图

        :param graph: 传入图
        :return graphic:返回经过networkx处理的关系图
        """
        graphic = nx.DiGraph()
        for node in range(0, len(graph) + 1):
            graphic.add_node(str(node))
        graphic.add_edges_from(graph)
        # fig, ax = plt.subplots()
        # nx.draw(graphic, ax=ax, with_labels=True)
        # plt.show()
        # print(graphic.degree)

        return graphic

    def removeGraphNode(self, graphic):
        """
        对图进行剪枝（移除出度为1，入度为0的图节点）

        :param graphic: 传入经过networkx处理的图
        :return:gCopy:返回经过剪枝后的图
        """
        # 出度
        outDegree = 1
        # 入度
        inDegree = 0
        gCopy = graphic.copy()
        gIn = gCopy.in_degree(gCopy)
        gOut = gCopy.out_degree(gCopy)
        nodeRemove = []
        for n in gCopy.nodes():
            if gIn[n] == inDegree and gOut[n] == outDegree:
                nodeRemove.append(n)
            if gIn[n] == 0 and gOut[n] == 0:
                nodeRemove.append(n)
        for node in nodeRemove:
            gCopy.remove_node(node)

        return gCopy

    def completeSentence(self, textStructure, nodes):
        """
        拼接关系产生新的句子

        :param textStructure:传入[代号, 文本, 词性, 关系指向, 关系]的数据结构
        :param nodes: 传入经过剪枝后的图
        :return: sentence:生成的句子
        """
        sentence = ''
        for text in textStructure:
            if str(text[0]) in nodes:
                sentence += text[1]

        return sentence

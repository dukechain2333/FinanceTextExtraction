#  - * -  coding:utf-8 - * -
import argparse
from FinanceText import FinanceText
import re
import jieba
from pyhanlp import HanLP
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QMainWindow
import FinExtractGUI_small
import sys

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--Source', type=str, default='FinText.txt', help='input path to read File')
    # parser.add_argument('--Target', type=str, default='FinExtract.txt', help='input path to save File')
    # parser.add_argument('--Preference', type=str, default='投资', help='input preference')
    #
    # opt = parser.parse_args()
    #
    # financeText = FinanceText(opt.Source, opt.Target, opt.Preference)
    # # financeText = FinanceText('FinText.txt', 'FinExtract.txt', '投资')
    # financeText.train_model()
    #
    # topic_words = []
    # sentence_words = []
    # topic_sentence = []
    # dic = {}
    # res = []
    # finalSentence = ''
    #
    # for line in financeText.text:
    #     line = re.sub(" ", "", line)
    #     line = re.sub("[].\\/_,$%^*(+\"\')]+|[+——（）【】、~@#￥%……&*（）“”《》]+", "", line)
    #     # topic_sentence.append(line)
    #     topic_words = jieba.lcut(line)
    #     # 建立词典统计切分的各个词的频率
    #     for word in topic_words:
    #         if word not in dic:
    #             dic[word] = 1
    #         else:
    #             dic[word] += 1
    #     pattern = r'\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|。|；|‘|’|【|】|·|！| |…|（|）'
    #     result_list = re.split(pattern, line)
    #
    #     topic_sentence.extend(result_list)
    #
    #     for i in range(len(topic_sentence)):
    #         score = 0
    #         for word in jieba.lcut(topic_sentence[i]):
    #             score += dic.get(word, 0)
    #         res.append(score)
    #     score_dic = {}
    #     for i in range(len(topic_sentence) - 1):
    #         temp_similarity = financeText.lda_sim(topic_words[i])
    #         score_dic[i] = (res[i] / (len(jieba.lcut(topic_sentence[i])) + 1)) * abs(temp_similarity)
    #     result = sorted(score_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    #     result1 = result[:2]
    #     for key in result1:
    #         # 输出原句
    #         # print(topic_sentence[key[0]])
    #         # 用HanLP进行分隔
    #         ans = HanLP.parseDependency(topic_sentence[key[0]].strip())
    #         # print(ans.toString())
    #         # 对分隔数据进行处理
    #         textStructure, graph = financeText.hanlp_split(ans.toString())
    #         # print(graph)
    #         # 生成词语关系图
    #         graphic = financeText.showGraph(graph)
    #         # 对关系图剪枝
    #         graphCut = financeText.removeGraphNode(graphic)
    #         # print(graphCut.nodes)
    #         # 拼凑新的句子
    #         sentence = financeText.completeSentence(textStructure, graphCut)
    #         finalSentence += sentence + '。'
    #         # print(sentence)
    #
    #     topic_words = []
    #     sentence_words = []
    #     topic_sentence = []
    #     dic = {}
    #     res = []
    #
    # print(finalSentence)

    # financeText.textWrite(finalSentence)
    myapp = QApplication(sys.argv)
    myDlg = QWidget()
    myUI = FinExtractGUI_small.Ui_widget()
    myUI.setupUi(myDlg)
    myDlg.setWindowIcon(QIcon('logo.jpg'))
    myDlg.show()
    sys.exit(myapp.exec_())

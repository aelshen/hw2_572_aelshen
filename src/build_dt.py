'''
#==============================================================================
build_dt.py
/Users/aelshen/Documents/Dropbox/School/CLMS 2013-2014/Winter 2014/Ling 571-Deep Processing Techniques for NLP/src/build_dt.py
Created on Jan 20, 2014
@author: aelshen
#==============================================================================
'''
import math
import os
import sys
import time
from collections import defaultdict
from collections import Counter
#==============================================================================
#--------------------------------Constants-------------------------------------
#==============================================================================
DEBUG = True
#==============================================================================
#-----------------------------------Main---------------------------------------
#==============================================================================
def main():
    
    start_time = time.time()
    
    training_data = sys.argv[1]
    test_data = sys.argv[2]
    max_depth = sys.argv[3]
    min_gain = sys.argv[4]
    model_file = sys.argv[5]
    sys_output = sys.argv[6]
    #acc_file = sys.argv[7]
    sys.setrecursionlimit(2000)    
    hw2 = DecisionTree(training_data, test_data, max_depth, min_gain,
                       model_file, sys_output)
    hw2.ImportTrainVectors()
    hw2.CreateTree(0, hw2.root)
    training_results = hw2.Test(hw2.train_file)
    testing_results = hw2.Test(hw2.test_file)
    hw2.PrintTree()
    hw2.PrintAcc(training_results, testing_results)
    
    elapsed_time = (time.time() - start_time)/60
    print "Elapsed time (in minutes): " + `elapsed_time`
#==============================================================================    
#---------------------------------Functions------------------------------------
#==============================================================================


#==============================================================================    
#----------------------------------Classes-------------------------------------
#==============================================================================
class DecisionTree:
    def __init__(self, train_file, test_file, max_depth, min_gain, model_file, 
                  sys_output):
        self.max_depth = int(max_depth)
        self.min_gain = float(min_gain)
        self.model_file = open(model_file, 'w')
        #opens a file briefly in order to empty it, then reopen it with 
        #privileges set to 'a'
        self.sys_output = open(sys_output, 'w').close()
        self.sys_output = open(sys_output, 'a')
        self.train_file = open(train_file, 'r')
        self.test_file = open(test_file, 'r')
        #self.acc_file = open(acc_file, 'w')
        self.root = None
        self.paths = dict()
    
    def ImportTrainVectors(self):
        feature_dict = defaultdict(list) 
        document_count = 0
        labels = defaultdict(int)
        for line in self.train_file:
            attributes = line.split()
            label = attributes[0]
            labels[label] += 1
            for attribute in attributes[1:]:
                feature = attribute.split(":")
                if feature[1] > 0:
                    feature_dict[feature[0]].append((label,document_count))
            #end for attribute in attributes[1:]
            document_count += 1
        #end for line in self.train_file
        
        self.root = Node(feature_dict, labels, document_count)
        self.root.featuer = "root"
        self.train_file.seek(0)
        
    def Descend(self, path, node):
        #if node == None:
        #    return
        
        if node.parent != None:
            path.append(node.feature)
            
        if node.terminal == True:
            temp = `node.document_count`
            for lbl in node.labels:
                prob = 0.0
                if node.labels[lbl] > 0:
                    prob = float(node.labels[lbl])/node.document_count
                temp += " " + lbl +" " + `prob`
            #end for lbl in node.labels
            self.paths["&".join(path)] = temp
            return
        self.Descend(list(path), node.feature_present)
        self.Descend(list(path), node.feature_absent)
            
    def PrintTree(self):
        self.Descend([], self.root)
        for path in self.paths:
            #self.model_file.write("&".join(path))
            self.model_file.write(path + " ")
            self.model_file.write(self.paths[path] + os.linesep)
        #for path in self.paths
        
        
    def CreateTree(self, depth, node):
        if depth > self.max_depth or node.document_count < 2:
            node.terminal = True
            return
        max_info_gain = 0.0
        max_feature = None
        
        for ft in node.features:
            feature_positive_entropy = 0.0
            feature_negative_entropy = 0.0
            
            positive_labels = defaultdict(int)
            negative_labels = defaultdict(int)
            for i in node.features[ft]:
                label = i[0]
                positive_labels[label] += 1
            #end for i in node.features[ft]
            
            for label in node.labels:
                negative_labels[label] = node.labels[label] - positive_labels[label]
            #for label in node.labels
            
            for lbl in node.labels:
                if lbl in positive_labels:
                    feature_positive_probability = positive_labels[lbl]/float(len(node.features[ft]))
                    if feature_positive_probability > 0:
                        feature_positive_entropy += feature_positive_probability * math.log(feature_positive_probability,2)
                if lbl in negative_labels:
                    feature_negative_probability = negative_labels[lbl]/float(len(node.features[ft]))
                    if feature_negative_probability > 0:
                        feature_negative_entropy += feature_negative_probability * math.log(feature_negative_probability,2)
            #for lbl in node.labels
            
            residual_entropy = (float( len(node.features[ft]) )/node.document_count) * feature_positive_entropy \
                               + (float( (node.document_count - len(node.features[ft]) ) )/node.document_count * feature_negative_entropy )
                               
            info_gain = node.entropy - residual_entropy

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_feature = ft
        #end for ft in self.features
        
        if max_info_gain <= self.min_gain or max_feature == None:
            node.terminal = True
            return
        
        next_document_count = len(node.features[max_feature])
        
        feature_positive = defaultdict(list)
        feature_negative = defaultdict(list)
        
        for feature in node.features:
            if feature == max_feature:
                continue
            else:
                for j in node.features[feature]:
                    if j in node.features[max_feature]:
                        feature_positive[feature].append(j)
                    else:
                        feature_negative[feature].append(j)
                #end for j in node.features[feature]
        #for feature in node.features
        
        positive_labels = defaultdict(int)
        negative_labels = defaultdict(int)
        for i in node.features[max_feature]:
            label = i[0]
            positive_labels[label] += 1
        #end for i in node.features[ft]
        
        for label in node.labels:
            negative_labels[label] = node.labels[label] - positive_labels[label]
            if label not in positive_labels:
                positive_labels[label] = 0 
        #for label in node.labels
        child_present = Node(feature_positive, positive_labels, next_document_count)
        child_present.feature = max_feature
        child_present.parent = node
        node.feature_present = child_present
        
        child_absent = Node(feature_negative, negative_labels, node.document_count - next_document_count)
        child_absent.feature = "!" + max_feature
        child_absent.parent = node
        node.feature_absent = child_absent
        
        self.CreateTree(depth + 1, child_present)
        self.CreateTree(depth + 1, child_absent)
    #end def DescendTree
    
    def Test(self, file):
        results = defaultdict(Counter)
        
        i = 0
        for line in file:
            self.sys_output.write("array" + `i` + ":\t")
            tokens = line.split()
            true_label = tokens[0]
            features = []
            for j in tokens[1:]:
                feature = j.split(":")
                features.append(feature[0])
            #for j in tokens[1:]
            cur_node = self.root
            while cur_node.terminal != True:
                if cur_node.feature_present.feature in features:
                    cur_node = cur_node.feature_present
                else:
                    cur_node = cur_node.feature_absent
            #end while cur_node.terminal != True
            
            dt_label = None
            dt_prob = 0.0
            
            for lbl in self.root.labels:
                prob = 0.0
                if cur_node.labels[lbl] > 0:
                    prob = float( cur_node.labels[lbl] ) / cur_node.document_count
                if prob > dt_prob:
                    dt_prob = prob
                    dt_label = lbl
                self.sys_output.write(lbl + "\t" + `prob` + "\t")
            #for lbl in self.root.labels
            self.sys_output.write(os.linesep)
            
            i += 1
            results[true_label][dt_label] += 1
        #for line in file
        return results

    def PrintAcc(self, training_results, testing_results):
        print "Confusion matrix for the training data:"
        print "row is the truth, column is the system output" + os.linesep
        
        labels = self.root.labels.keys()
        
        correct_labels = 0
        total_labels = 0
        
        print "\t"*3,
        for lbl in labels:
            print lbl + "\t",
        #for lbl in labels
        print ""
        
        for i in labels:
            print i + "\t",
            for j in labels:
                if i == j:
                    correct_labels += training_results[i][j]
                total_labels += training_results[i][j]
                print `training_results[i][j]` + "\t",   
            #end for j in labels
            print ""
        #end for i in labels
        
        accuracy = float(correct_labels)/total_labels
        print os.linesep + "Training Accuracy = " + `accuracy` + os.linesep*2
        
        correct_labels = 0
        total_labels = 0
        
        print "\t"*3,
        for lbl in labels:
            print lbl + "\t",
        #for lbl in labels
        print ""
        
        for i in labels:
            print i + "\t",
            for j in labels:
                if i == j:
                    correct_labels += testing_results[i][j]
                total_labels += testing_results[i][j]
                print `testing_results[i][j]` + "\t", 
            #end for j in labels
            print ""
        #end for i in labels
        
        accuracy = float(correct_labels)/total_labels
        print os.linesep + "Test Accuracy = " + `accuracy` + os.linesep*2
        
        
    
    def __exit__(self, type, value, traceback):
        self.train_file.close()
        self.test_file.close()
        self.model_file.close()
        self.sys_output.close()
        #self.acc_file.close()

class Node:
    def __init__(self, features, labels, document_count):
        self.document_count = document_count
        self.feature = None
        self.features = features
        self.labels = labels
        self.entropy = self.CalculateEntropy()
        self.parent = None
        self.feature_present = None
        self.feature_absent = None
        self.terminal = False
    
    def CalculateEntropy(self):
        entropy = 0.0
        for lbl in self.labels:
            if self.labels[lbl] > 0:
                prob = float(self.labels[lbl])/self.document_count
                entropy -= prob * math.log(prob,2)
        #end for lbl in self.labels
        return entropy


if __name__ == "__main__":
    sys.exit( main() )

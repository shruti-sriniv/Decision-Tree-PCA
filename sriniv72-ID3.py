##############
# Name: Shruti Srinivasan
# email: sriniv72@purdue.edu
# Date: 10/26/2020

import numpy as np
import pandas as pd

def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy


def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, l, r, attr, thresh, label):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.label = label


class Tree(object):
    def __init__(self, ____):
        pass

def findMax(train_labs):
    countZ = 0
    countO = 0
    for i in train_labs:
        if i == 0:
            countZ += 1
        else:
            countO += 1
    if countZ > countO:
        return 0
    else:
        return 1

def makeSureDead(dat):
    if 0 not in dat:
        dat[0] = 0
    if 1 not in dat:
        dat[1] = 0
    return dat


def findLabel(train_labels):
    zeroes = 0
    ones = 0
    for lab in train_labels:
        if lab == 0:
            zeroes += 1
        else:
            ones += 1

    if ones > zeroes:
        return 1
    else:
        return 0

def valuCounts(data):
    return data['survived'].value_counts()


def calculate_thresh(train_data, attribute):
    column = train_data[attribute]
    dict = {}
    thresh = 0
    infor = 0
    uniq = column.unique().tolist()
    uniq.sort()

    for i in range(0, len(uniq) - 1):
        threshold = (uniq[i] + uniq[i + 1]) * 0.5

        if threshold != threshold:
            continue
        left_side_split = train_data[train_data[attribute] < threshold]
        right_side_split = train_data[train_data[attribute] >= threshold]
        if left_side_split.empty or right_side_split.empty:
            continue
        left = makeSureDead(valuCounts(left_side_split))
        right = makeSureDead(valuCounts(right_side_split))

        before_split_freq = [makeSureDead(valuCounts(train_data))[0], makeSureDead(valuCounts(train_data))[1]]
        after_split_freq = [[left[0], left[1]], [right[0], right[1]]]
        info = infor_gain(before_split_freq, after_split_freq)


        dict.update({threshold: info})

        all_values = dict.values()
        thresh = max(dict, key=dict.get)
        infor = max(all_values)

    return [thresh, infor]



def ID3(train_data, train_labels, spl):
    the_chosen_threshold = 0
    the_chosen_info = 0
    the_chosen_attribute = ""
    the_chosen_ones = {}
    total = pd.concat([train_data, train_labels], axis=1, ignore_index=False)
    if len(total['survived'].unique()) == 1:
        return Node(None, None, None, None, findLabel(train_labels))

    for att in total:
        if att != 'survived':
            threshold = calculate_thresh(total, att)[0]
            ideal_gain = calculate_thresh(total, att)[1]

            the_chosen_ones.update({(att, threshold): ideal_gain})
            all_values = the_chosen_ones.values()
            values = max(the_chosen_ones, key=the_chosen_ones.get)
            the_chosen_info = max(all_values)

            the_chosen_threshold = values[1]
            the_chosen_attribute = values[0]

    if the_chosen_info != 0:
        current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, findLabel(train_labels))

        left_part_train_data = total[total[the_chosen_attribute] < the_chosen_threshold]
        right_part_train_data = total[total[the_chosen_attribute] >= the_chosen_threshold]

        left_label = left_part_train_data['survived']
        right_label = right_part_train_data['survived']
        left_part_train_data = left_part_train_data.drop(columns=['survived', the_chosen_attribute], axis=1)
        right_part_train_data = right_part_train_data.drop(columns=['survived', the_chosen_attribute], axis=1)

        if len(left_part_train_data) < spl or len(right_part_train_data) < spl:
            return Node(None, None, None, None, findLabel(train_labels))

        # print(current_node.attribute)
        left_subtree = ID3(left_part_train_data, left_label, spl)
        right_subtree = ID3(right_part_train_data, right_label, spl)
        current_node.left_subtree = left_subtree
        current_node.right_subtree = right_subtree
        return current_node

    else:
        return Node(None, None, None, None, findLabel(train_labels))

train_data = pd.read_csv('titanic-train.data',index_col=None, engine='python')
train_label = pd.read_csv('titanic-train.label',index_col=None, engine='python')

# ID3(train_data, train_label, 0)
# node = ID3(train_data, train_label)
# print(node.left_subtree, node.right_subtree, node.attribute, node.threshold)

def predict2(tree, j):
    if tree.left_subtree == None and tree.right_subtree == None:
        return tree.label
    if tree.threshold > j[tree.attribute]:
        return predict2(tree.left_subtree, j)
    if tree.threshold <= j[tree.attribute]:
        return predict2(tree.right_subtree, j)
    if tree.attribute == None and tree.threshold == None:
        return tree.label


def predict(tree, train_data, train_label):
    x = 0
    y = 0
    for i, j in train_data.iterrows():
        if predict2(tree, j) == train_label['survived'][index]:
            x += 1
        y += 1
    return x / len(train_data)


def resizeTree(tree, depth):
    if depth == 1:
        tree.left_subtree = None
        tree.right_subtree = None
        return tree
    elif tree is None:
        return 0
    else:
        resizeTree(tree.left_subtree, depth)
        resizeTree(tree.right_subtree, depth)


"""
class PCA(object):
     def __init__(self, n_component):
         self.n_component = n_component

def fit_transform(self, train_data):
[TODO] Fit the model with train_data and
apply the dimensionality reduction on train_data]

def transform(self, test_data):
[TODO Apply dimensionality reduction to test_data]


a = np.arange(8.0)
splits = np.array_split(a, 3)
for i in splits:
    print(i)
    """
__name__ = "__main__"

def removeNone(test_list):
    res = []
    for val in test_list:
        for values in val:
            if values != None:
                res.append(values)
    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder', dest="train")
    parser.add_argument('--testFolder', dest="test")
    parser.add_argument('--model', dest="model")
    parser.add_argument('--crossValidK', type=int, default=5, dest="K")
    parser.add_argument('--minSplit', type=int, default=0, dest='min_split')
    parser.add_argument('--depth', type=int, default=10, dest='depth')
    args = parser.parse_args()
    print(args)

    train_data = pd.read_csv(args.train, delimiter=',', index_col=None, engine='python')
    test_data = pd.read_csv(args.test, delimiter=',', index_col=None, engine='python')

    train_subs = np.array_split(train_data, args.K)
    labs = np.array_split(test_data, args.K)
    split = 0
    for index in range(0, len(train_subs)):
        validation_input = train_subs[index].reset_index(drop=True)
        validation_labels = labs[index].reset_index(drop=True)

        training_input = train_subs.copy()
        training_input[index] = None
        training_input = pd.concat(training_input).reset_index(drop=True)
        training_labels = labs.copy()
        training_labels[index] = None
        training_labels = pd.concat(training_labels).reset_index(drop=True)

        model = ID3(training_input, training_labels, 0)
        if args.model == "minSplit":
            model = ID3(training_input, training_labels, args.min_split)

        if args.model == "depth":
            resizeTree(model, args.depth)

        print("fold=" + str(index + 1) + ", Train set accuracy= "
              + str(predict(model, training_input, training_labels)) +
              ", Validation set accuracy= " + str(predict(model, validation_input, validation_labels)))










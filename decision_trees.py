import numpy as np

X = np.array([[0,1,0,1,1],[0,0,0,1,1],[1,0,1,1,0],[0,0,0,0,1]])
Y = np.array([[1],[0],[0],[1]])
Y2 = np.array([[0],[1],[0],[1]])
max_depth = -1

X_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1],
                      [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
Y_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

## Decision Trees
# Training Set 1:
X_train_1 = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]])
Y_train_1 = np.array([[1], [0], [0], [0], [1]])
# Validation Set 1:
X_val_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_val_1 = np.array([[0], [1], [0], [1]])
# Testing Set 1:
X_test_1 = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_test_1 = np.array([[1], [1], [0], [1]])
# Training Set 2:
X_train_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1],
                      [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
Y_train_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
# Validation Set 2:
X_val_2 = np.array([[1,0,0,0], [0,0,1,1], [1,1,0,1],
                    [1,1,0,0], [1,0,0,1], [0,1,0,0]])
Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
# Testing Set 2:
X_test_2 = np.array([[0,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1], [1,1,0,1],
                     [1,1,0,0], [1,0,0,1], [0,1,0,1], [0,1,0,0]])
Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])


class Node:
    def __init__(self, root=False, feature=-1, accuracy=0, samples=[], label=0, convergence=False, left=None, right=None):
        self.root = False
        self.feature = feature
        self.left = left
        self.right = right
        self.accuracy = accuracy
        self.samples = samples
        self.label = label
        self.convergence = convergence

    def getAccuracy(self):
        return self.accuracy

    def setAccuracy(self, newAccuracy):
        self.accuracy = newAccuracy

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def __iter__(self):
        if self.left != None:
            for elem in self.left:
                yield elem

        yield self.accuracy

        if self.right != None:
            for elem in self.right:
                yield elem

    def __repr__(self):
        return "decisionTree.Node(" + repr(self.accuracy) + "," + repr(self.left) + "," + repr(self.right) + ")"

class decisionTree:
    # decisionTree methods
    def __init__(self, root=None, left=None, right=None):
        self.root = root

    def insert(self, leftOrRight, feature, accuracy, samples, label):
        if(leftOrRight == "left"):
            if self.root.left == None:
                self.root.left = Node(False, feature, accuracy, samples, label)
        else:
            if self.root.right == None:
                self.root.right = Node(False, feature, accuracy, samples, label)

    def returnAccuracy(self):
        return self.root.left.getAccuracy + self.root.right.getAccuracy

    def printTree(self):
        print "Root: "
        print self.root.feature
        print self.root.samples
        print self.root.accuracy
        print "Left"
        print self.root.left.feature
        print self.root.left.samples
        print self.root.left.accuracy
        print "Right: "
        print self.root.right.feature
        print self.root.right.samples
        print self.root.right.accuracy
        print "Left Left Children"
        print self.root.left.left.feature
        print self.root.left.left.samples
        print self.root.left.left.accuracy
        print self.root.left.left.left.accuracy
        print "Left Right Children"
        print self.root.left.right.feature
        print self.root.left.right.samples
        print self.root.left.right.accuracy


def labelCount(L, labels, samples):
    count = 0
    for i in range(labels.shape[0]):
        if labels[i][0] == L and i in samples:
            count += 1
    return count

def calcAccuracy(zeroCount, oneCount):
    if zeroCount > oneCount:
        return float(zeroCount) / float(oneCount+zeroCount)
    elif oneCount > zeroCount:
        return float(oneCount) / float(zeroCount+oneCount)
    else:
        return .5

def returnLabel(zeroCount, oneCount):
    if zeroCount > oneCount:
        return 0
    elif oneCount > zeroCount:
        return 1
    else:
        return 1

def depthOneDT(samples, usedFeatures,X, Y):
    accuracyMax = -1
    for col in range(X.shape[1]):
        if col not in usedFeatures:
            zeroSamples = []
            oneSamples = []
            for row in range(X.shape[0]):
                if row in samples:
                    if X[row][col] == 0:
                        zeroSamples.append(row)
                    else:
                        oneSamples.append(row)
            #get number of Y and N in Y corresponding to zeroSamples and oneSamples indices
            oneCountLeft = labelCount(1, Y, zeroSamples)
            zeroCountLeft = labelCount(0, Y, zeroSamples)
            oneCountRight = labelCount(1, Y, oneSamples)
            zeroCountRight = labelCount(0, Y, oneSamples)

            accuracy = calcAccuracy(zeroCountLeft, oneCountLeft) + calcAccuracy(zeroCountRight, oneCountRight)
            if accuracy > accuracyMax:
                #print accuracy
                accuracyMax = accuracy
                root = Node(True, col)
                tree = decisionTree(root)
                tree.root.setAccuracy(accuracy)
                tree.root.samples = zeroSamples + oneSamples
                tree.insert("left", -1, calcAccuracy(zeroCountLeft, oneCountLeft), zeroSamples, returnLabel(zeroCountLeft, oneCountLeft))
                tree.insert("right", -1, calcAccuracy(zeroCountRight, oneCountRight), oneSamples, returnLabel(zeroCountRight, oneCountRight))
    return tree


def DT_train_binary(X, Y, max_depth):
    #print "MAX DEPTH!"
    #print max_depth
    if max_depth == 1:
        return

    flag = True
    usedFeatures = []
    # get initial depth 2 tree
    tree = depthOneDT(range(X.shape[0]),usedFeatures,X,Y)
    depth = tree.root.depth()
    if(depth >= max_depth and max_depth != -1):
        return tree
    else:
        # call recursive tree building function on root
        DT_recursive(tree.root,tree.root,usedFeatures,X,Y,depth,max_depth)
    return tree

def DT_recursive(mainroot, root, usedFeatures, X, Y, depth, max_depth):
    #base case
    if root.left.convergence and root.right.convergence:
        return True

    if len(root.left.samples) == 1 or root.left.accuracy == 1:
        root.left.convergence = True
    if len(root.right.samples) == 1 or root.right.accuracy == 1:
        root.right.convergence = True

    # keep track of features that have already been used
    usedFeatures.append(root.feature)

    if (not root.left.convergence) and len(usedFeatures) != X.shape[1]:
        # call depthOneDT on left branch
        newLeft = depthOneDT(root.left.samples, usedFeatures, X,Y)
        # check if the accuracy has increased, if so, replace with better tree
        if newLeft.root.accuracy > root.left.accuracy:
            root.left = newLeft.root
    else:
        #else branch has converged
        #print "converged!"
        root.left.convergence = True
    if (not root.right.convergence) and len(usedFeatures) != X.shape[1]:
        newRight = depthOneDT(root.right.samples, usedFeatures, X,Y)
        if newRight.root.accuracy > root.right.accuracy:
            root.right = newRight.root
    else:
        root.right.convergence = True

    if mainroot.depth() == max_depth and max_depth != -1:
        return True

    if (not root.right.convergence):
        return DT_recursive(root, root.right, usedFeatures, X, Y, depth, max_depth)
    if (not root.left.convergence):
        #print "RECURSIVE CALL TO LEFT"
        return DT_recursive(root, root.left, usedFeatures, X, Y, depth, max_depth)
    return True

def DT_test_binary(X,Y,DT):
    correct = 0
    #for every sample in test data
    for i in range(X.shape[0]):
        correct += DT_test_recursive(X,Y,DT.root,i)
    return float(correct) / float(X.shape[0])

def DT_test_recursive(X,Y,root,sample):
    if root.feature != -1:
        if X[sample][root.feature] == 0:
            return DT_test_recursive(X,Y,root.left,sample)
        else:
            return DT_test_recursive(X,Y,root.right,sample)
    else:
        if root.label == Y[sample]:
            return 1
        else:
            return 0

def DT_train_binary_best(X_train,Y_train,X_val,Y_val):
    flag = True
    max_depth = 2
    tree = DT_train_binary(X_train,Y_train,max_depth)
    accuracy = DT_test_binary(X_val,Y_val,tree)
    while flag:
        max_depth += 1
        if accuracy < DT_test_binary(X_val,Y_val,DT_train_binary(X_train,Y_train,max_depth)):
            tree = DT_train_binary(X_train,Y_train,max_depth)
            accuracy = tree.root.accuracy
            #print "accuracy: "
            #print accuracy
        else:
            return tree
'''
print "TRIAL #1"
tree = DT_train_binary(X_train_1, Y_train_1, max_depth)
accuracy = DT_test_binary(X_test_1,Y_test_1,tree)
#accuracy = DT_test_binary(X_train_1, Y_train_1,tree)
print "accuracy1"
print accuracy
treeBest = DT_train_binary_best(X_train_1,Y_train_1,X_val_1,Y_val_1)
accuracy2 = DT_test_binary(X_test_1,Y_test_1,treeBest)
print "accuracy2"
print accuracy
print " "

print "TRIAL #2"
tree2 = DT_train_binary(X_train_2, Y_train_2, max_depth)
print tree2.root.accuracy
print tree2.root.depth()
accuracy = DT_test_binary(X_test_2,Y_test_2,tree2)
#accuracy = DT_test_binary(X_train_2, Y_train_2,tree2)
print "accuracy1"
print accuracy
treeBest2 = DT_train_binary_best(X_train_2,Y_train_2,X_val_2,Y_val_2)
print treeBest2.root.accuracy
print "treeBest2 depth"
print treeBest2.root.depth()
accuracy2 = DT_test_binary(X_test_2,Y_test_2,treeBest2)
print "accuracy2"
print accuracy2'''

import numpy

class decisionTree:
    class Node:
        def __init__(self, root, feature, left=None, right=None, accuracy=0, samples):
            self.root = False
            self.feature = feature
            self.left = left
            self.right = right
            self.accuracy = accuracy
            self.samples = samples

        def getAccuracy(self):
            return self.accuracy

        def setAccuracy(self, newAccuracy):
            self.accuracy = newAccuracy

        def getLeft(self):
            return self.left

        def getRight(self):
            return self.right

        def setLeft(self, newLeft):
            self.left = newLeft

        def setRight(self, newRight):
            self.right = newRight

        def setMainRoot():
            self.root = True

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


        # decisionTree methods
        def __init__(self, root=None):
            self.root = root

        def insert(self, feature, leftOrRight, accuracy):
            if(leftOrRight == "left"):
                if self.left == None:
                    self.left = Node(False, feature, accuracy)
            else:
                if self.right == None:
                    self.right = Node(False, feature, accuracy)

        def returnAccuracy(self):
            return self.left.getAccuracy + self.right.getAccuracy


def DT_train_binary(X, Y, max_depth):
    max_accuracy = 0
    for i in range(X.shape[1]):
        root = Node(True, i)
        tree = decisionTree(root)
        numZero = 0
        numOne = 0
        for j in range(X[i]):
            if X[i][j] == 0:
                numZero += 1
                zeroSamples.append(j)
            else:
                numOne += 1
                oneSamples.append(j)
        #get number of Y and N in Y corresponding to zeroSamples and oneSamples indices
        oneCount = Y.count(1)
        zeroCount = Y.count(0)

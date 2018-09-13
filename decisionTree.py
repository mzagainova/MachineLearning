class decisionTree:
    class Node:
        def __init__(self, feature, left=None, right=None, accuracy=0):
            self.feature = feature
            self.left = left
            self.right = right
            self.accuracy = accuracy

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

        def insert(root, feature)

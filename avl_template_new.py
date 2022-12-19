# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info

from random import randint

"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

	@type value: str
	@param value: data of your node
	"""

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1  # height of the node
        self.size = 0  # number of nodes in the left and right subtrees + 1
        self.bf = 0  # balance factor of the node

    def recc(self):
        if (self.isRealNode() == True):
            if abs(self.getBF()) > 2:
                return False
            self.getLeft().recc()
            self.getRight().recc()
        return True

    """returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""

    def getLeft(self):
        return self.left

    """returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""

    def getRight(self):
        return self.right

    """returns the parent 

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

    def getParent(self):
        return self.parent

    """return the value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""

    def getValue(self):
        return self.value

    """returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

    def getHeight(self):
        return self.height

    """returns the size

	@rtype: int
	@returns: the size of self, 0 if node is virtual
	"""

    def getSize(self):
        return self.size

    """returns the balance factor

	@rtype: int
	@returns: the balance factor of self
	"""

    def getBF(self):
        return self.bf

    """sets left child

	@type node: AVLNode
	@param node: a node
	"""

    def setLeft(self, node):
        self.left = node

    """sets right child

	@type node: AVLNode
	@param node: a node
	"""

    def setRight(self, node):
        self.right = node

    """sets parent

	@type node: AVLNode
	@param node: a node
	"""

    def setParent(self, node):
        self.parent = node

    """sets value

	@type value: str
	@param value: data
	"""

    def setValue(self, value):
        self.value = value

    """sets the height

	@type h: int
	@param h: data
	"""

    def setHeight(self, h):
        self.height = h

    """sets the size

	@type s: int
	@param s: data
	"""

    def setSize(self, s):
        self.size = s

    """sets the balance factor

	@type bf: int
	@param bf: data
	"""

    def setBF(self, bf):
        self.bf = bf

    """returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

    def isRealNode(self):
        if self.height == -1:
            return False
        return True

    """computes the balance factor

	@rtype: int
	@returns: the balance factor that self should have, i.e height(left_subtree) - height(right_subtree)
	"""

    def computeBF(self):
        return (1 + self.getLeft().getHeight()) - (1 + self.getRight().getHeight())

    """computes the height

	@rtype: int
	@returns: the height that self should have, i.e 1 + max(height(left_subtree), height(right_subtree))
	"""

    def computeHeight(self):
        return 1 + max(self.getLeft().getHeight(), self.getRight().getHeight())

    """computes the size

	@rtype: int
	@returns: the size that self should have, i.e 1 + size(left_subtree) + size(right_subtree)
	"""

    def computeSize(self):
        return 1 + self.getLeft().getSize() + self.getRight().getSize()


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
	Constructor, you are allowed to add more fields.

	"""

    def __init__(self):
        self.size = 0
        self.root = AVLNode(None)
        self.min = None  # left most node in the tree/min in list
        self.max = None  # right most node in the tree/max in list

    def isAvlTree(self):
        if self.getRoot() == None:
            return True
        return self.getRoot().recc()

    """returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, virtual node if the list is empty
	"""

    def getRoot(self):
        return self.root

    """returns the maximum node of the tree representing the list

	@rtype: AVLNode
	@returns: the right most node, None if the list is empty
	"""

    def getMax(self):
        return self.max

    """returns the minimum node of the tree representing the list

	@rtype: AVLNode
	@returns: the left most node, None if the list is empty
	"""

    def getMin(self):
        return self.min

    """sets the size of the tree representing the list

	@type s: int
	@param s: data
	"""

    def setSize(self, s):
        self.size = s

    """sets the root of the tree representing the list

	@type node: AVLNode
	@param node: a node
	"""

    def setRoot(self, node):
        self.root = node

    """sets the max of the tree representing the list

	@type node: AVLNode
	@param node: a node
	"""

    def setMax(self, node):
        self.max = node

    """sets the min of the tree representing the list

	@type node: AVLNode
	@param node: a node
	"""

    def setMin(self, node):
        self.min = node

    """returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""

    def empty(self):
        if not self.getRoot().isRealNode():
            return True
        return False

    """retrieves the value of the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the value of the i'th item in the list
	"""

    def retrieve(self, i):
        return self.treeSelect(i + 1).getValue()

    """inserts val at position i in the list

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

    def insert(self, i, val):
        # initializing a node to insert into the tree
        node_to_insert = AVLNode(val)
        node_to_insert.setRight(AVLNode(None))
        node_to_insert.setLeft(AVLNode(None))
        node_to_insert.setHeight(node_to_insert.computeHeight())
        node_to_insert.setSize(node_to_insert.computeSize())

        # initialize a non-empty tree
        if self.empty():
            node_to_insert.setParent(self.getRoot())
            self.setRoot(node_to_insert)
            self.setSize(self.length() + 1)
            self.setMax(node_to_insert)
            self.setMin(node_to_insert)
            return 0

        # faster access to insertion in the end of the tree representing a list
        if i == self.length():
            node = self.getMax()
            node.setRight(node_to_insert)
            self.setMax(node_to_insert)

        # faster access to insertion in the start of the tree representing a list
        elif i == 0:
            node = self.getMin()
            node.setLeft(node_to_insert)
            self.setMin(node_to_insert)

        # inserting a new node in the center of the tree representing a list
        else:
            # inserting the new node as the predecessor to the (i+1)'th smallest node in the tree representing a list
            node = self.treeSelect(i + 1)
            if not node.getLeft().isRealNode():
                node.setLeft(node_to_insert)
            else:
                node = self.predecessor(node)
                node.setRight(node_to_insert)

        node_to_insert.setParent(node)
        self.setSize(self.length() + 1)

        # rebalance the tree after insertion
        num_rebalancing = self.rebalanceTree(node)
        return num_rebalancing

    """deletes the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

    def delete(self, i):

        # if the current tree has only one node in it, update it to an empty tree
        if self.length() == 1:
            self.setRoot(AVLNode(None))
            self.setSize(0)
            self.setMin(None)
            self.setMax(None)
            return 0

        # preparing deletion from the end of the tree representing a list
        if i == (self.length() - 1):
            node_to_del = self.getMax()
        # print("node to del: " + str(node_to_del.getValue()))
        # new_max = self.predecessor(node_to_del)
        # print("predecessor: " + str(new_max.getValue()))
        # self.setMax(new_max)
        # print("new max: " + str(self.last()))

        # preparing deletion from the start of the tree representing a list
        elif i == 0:
            node_to_del = self.getMin()
            self.setMin(successor(node_to_del))

        # preparing deletion from the center of the tree representing a list
        else:
            node_to_del = self.treeSelect(i + 1)

        # deleting the node from the tree
        node_to_rebalance = self.delete_node(node_to_del)

        # rabalancing the tree
        num_rebalancing = self.rebalanceTree(node_to_rebalance)
        self.setSize(self.length() - 1)
        new_max = self.treeSelect(self.length())
        self.setMax(new_max)
        return num_rebalancing

    """deletes a certain node from the tree representing the list

	@type node_to_del: AVLNode
	@param node_to_del: the node that will be deleted from the tree 
	@pre: node_to_del in self
	@rtype: AVLNode
	@returns: the node from which we need to start rebalancing
	"""

    def delete_node(self, node_to_del):
        # deleting node with no child
        if not node_to_del.getLeft().isRealNode() and not node_to_del.getRight().isRealNode():
            # print("node has no children: " + str(node_to_del.getValue()))
            parent = node_to_del.getParent()
            if parent.getLeft() is node_to_del:
                parent.setLeft(node_to_del.getLeft())
            else:
                parent.setRight(node_to_del.getLeft())
            return parent

        # deleting node with only right child
        elif not node_to_del.getLeft().isRealNode():
            # print("node has right child: " + str(node_to_del.getValue()))
            right_son = node_to_del.getRight()
            parent = node_to_del.getParent()
            if parent.isRealNode():
                if parent.getLeft() is node_to_del:
                    parent.setLeft(right_son)
                else:
                    parent.setRight(right_son)
            else:
                self.setRoot(right_son)
            right_son.setParent(parent)
            return right_son

        # deleting node with only left child
        elif not node_to_del.getRight().isRealNode():
            # print("node has left child: " + str(node_to_del.getValue()))
            left_son = node_to_del.getLeft()
            parent = node_to_del.getParent()
            if parent.isRealNode():
                if parent.getLeft() is node_to_del:
                    parent.setLeft(left_son)
                else:
                    parent.setRight(left_son)
            else:
                self.setRoot(left_son)
            left_son.setParent(parent)
            return left_son

        # Node with two children:
        else:
            # Get the inorder successor and copy it's value to this node
            suc_node = successor(node_to_del)
            node_to_del.setValue(suc_node.getValue())

            # Delete the inorder successor
            return self.delete_node(suc_node)

    """returns the value of the first item in the list

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""

    def first(self):
        if not self.getRoot().isRealNode():
            return None
        return self.getMin().getValue()

    """returns the value of the last item in the list

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""

    def last(self):
        if not self.getRoot().isRealNode():
            return None
        return self.getMax().getValue()

    """returns an array representing list 

	@rtype: list
	@returns: a list of strings representing the data structure
	"""

    def listToArray(self):
        L = []
        self.inorder(self.getRoot(), L)
        return L

    """appends the nodes of the tree in-order to the list 

	@type node: AVLNode
	@type L: list
	@param node: a node from which to start appending in-order
	@param L: the list representing the tree
	"""

    def inorder(self, node, L):
        if node.isRealNode():
            # get values in-order in left subtree
            self.inorder(node.getLeft(), L)
            # append the value of the node to the list representing the tree
            L.append(node.getValue())
            # get values in-order in right subtree
            self.inorder(node.getRight(), L)

    """returns the size of the list 

	@rtype: int
	@returns: the size of the list
	"""

    def length(self):
        return self.getRoot().getSize()

    """sort the info values of the list

	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	"""

    def sort(self):
        # get the list that represents the tree
        tree_arr = self.listToArray()

        # create a sorted copy of the list
        org_arr = merge_sort(tree_arr)

        # create new tree using the sorted list
        new_tree = AVLTreeList()
        new_tree.root = buildTreeFromList(org_arr, 0, len(org_arr) - 1)
        new_tree.size = new_tree.root.getSize()
        new_tree.max = org_arr[len(org_arr) - 1]
        new_tree.min = org_arr[0]
        new_tree.getRoot().setParent(AVLNode(None))
        return new_tree

    """permute the info values of the list 

	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""

    def permutation(self):
        # get the list that represents the tree
        tree_arr = self.listToArray()

        # shuffle the list in place
        shuffle(tree_arr)

        # create new tree using the shuffled list
        new_tree = AVLTreeList()
        new_tree.root = buildTreeFromList(tree_arr, 0, len(tree_arr) - 1)
        new_tree.size = new_tree.root.getSize()
        new_tree.max = tree_arr[len(tree_arr) - 1]
        new_tree.min = tree_arr[0]
        new_tree.getRoot().setParent(AVLNode(None))
        return new_tree

    """concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

    def concat(self, lst):
        lst_height = lst.getRoot().getHeight()
        self_height = self.getRoot().getHeight()

        # the other tree is an empty tree
        if lst.empty():
            return abs(self_height - lst_height)

        # self is an empty tree
        if self.empty():
            self.setRoot(lst.getRoot())
            self.min = lst.min
            self.max = lst.max
            self.size = lst.size
            return abs(self_height - lst_height)

        # joining the non-empty trees
        self.max.setRight(lst.getRoot())
        lst.getRoot().setParent(self.max)
        self.max = lst.max
        self.rebalanceTree(self.max)
        self.size = self.getRoot().getSize()
        return abs(self_height - lst_height)

    """searches for a *value* in the list

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""

    def search(self, val):
        arr = self.listToArray()
        for i, value in enumerate(arr):
            if val == value:
                return i
        return -1

    """performs a right rotation on the node

	@type node: AVLNode
	@param val: a node to rotate right
	"""

    def rotateRight(self, node):
        A = node.getLeft()
        node.setLeft(A.getRight())
        A.getRight().setParent(node)
        A.setRight(node)
        A.setParent(node.getParent())

        # node is not the root of the tree
        if node.getParent().isRealNode():
            if node.getParent().getLeft() == node:
                A.getParent().setLeft(A)
            else:
                A.getParent().setRight(A)

        # node is the root of the tree
        else:
            self.setRoot(A)

        # updating the swapped nodes parameters
        node.setParent(A)
        node.setHeight(node.computeHeight())
        A.setHeight(A.computeHeight())
        A.setSize(node.getSize())
        node.setSize(node.computeSize())
        node.setBF(node.computeBF())
        A.setBF(A.computeBF())

    """performs a left rotation on the node

	@type node: AVLNode
	@param val: a node to rotate left
	"""

    def rotateLeft(self, node):
        A = node.getRight()
        node.setRight(A.getLeft())
        A.getLeft().setParent(node)
        A.setLeft(node)
        A.setParent(node.getParent())

        # node is not the root of the tree
        if node.getParent().isRealNode():
            if node.getParent().getLeft() == node:
                A.getParent().setLeft(A)
            else:
                A.getParent().setRight(A)

        # node is the root of the tree
        else:
            self.setRoot(A)

        # updating the swapped nodes parameters
        node.setParent(A)
        node.setHeight(node.computeHeight())
        A.setHeight(A.computeHeight())
        A.setSize(node.getSize())
        node.setSize(node.computeSize())
        node.setBF(node.computeBF())
        A.setBF(A.computeBF())

    """rebalances the tree representing the list

	@type node: AVLNode
	@param val: the node to start reblancing from upwards
	@rtype: int
	@returns: the number of rotations during the rebalancing process
	"""

    def rebalanceTree(self, node):
        num_rebalancing = 0
        while node.isRealNode():
            # update parameters of the node
            node.setHeight(node.computeHeight())
            node.setSize(node.computeSize())
            node.setBF(node.computeBF())

            # rebalance the tree
            if abs(node.getBF()) >= 2:
                # as long as the current node is not balanced (happens more than one time during joining lists)
                while abs(node.getBF()) >= 2:
                    # right subtree is too big
                    if node.getBF() <= -2:
                        right_son = node.getRight()
                        if right_son.getBF() == 1:
                            AVLTreeList.rotateRight(self, node.getRight())
                            AVLTreeList.rotateLeft(self, node)
                            num_rebalancing += 2
                        else:
                            AVLTreeList.rotateLeft(self, node)
                            num_rebalancing += 1

                    # left subtree is too big
                    else:
                        left_son = node.getLeft()
                        if left_son.getBF() == -1:
                            AVLTreeList.rotateLeft(self, node.getLeft())
                            AVLTreeList.rotateRight(self, node)
                            num_rebalancing += 2
                        else:
                            AVLTreeList.rotateRight(self, node)
                            num_rebalancing += 1
                    node.setBF(node.computeBF())
            node = node.getParent()
        return num_rebalancing

    """finds and returns the node in index i - 1 of the list

	@type i: int
	@pre: 0 < i <= self.length()
	@param i: the i'th smallest element in the tree
	@rtype: AVLNode
	@returns: the i'th node of the tree representing the list
	"""

    def treeSelect(self, i):
        node = self.getMin()

        # locate the node that has at least i children
        while node.getSize() < i:
            node = node.getParent()
        return self.treeSelectHelper(node, i)

    """a helper function to treeSelect

	@type node: AVLNode
	@type i: int
	@param node: a node
	@param i: the i'th smallest element in the tree
	@pre 0 <= i < self.length()
	@rtype: AVLNode
	@returns: the i'th node in the tree representing the list
	"""

    def treeSelectHelper(self, node, i):
        r = node.getLeft().getSize() + 1
        # node is the i'th smallest element in the tree
        if i == r:
            return node

        # the node we are searching for is in the left subtree
        elif i < r:
            return self.treeSelectHelper(node.getLeft(), i)

        # the node we are searching for is in the right subtree
        else:
            return self.treeSelectHelper(node.getRight(), i - r)

    """finds the predecessor node of x in the tree

	@type x: AVLNode
	@param x: a node
	@rtype: AVLNode
	@returns: the predecessor of x
	"""

    def predecessor(self, x):
        # the node has a left subtree and therefore the predecessor of x is the max node in the left subtree
        print("x: " + str(x.height))
        print("x: " + str(x.value))
        print("is x left node real: " + str(x.getLeft().isRealNode()))
        if x.getLeft().isRealNode():
            return max_node(x.getLeft())

        # find the lowest ancestor y such that x is in the right subtree of y
        y = x.getParent()
        while y.isRealNode() and x == y.getLeft():
            x = y
            y = x.getParent()
        return y

    ##########################################
    ## This file contains functions for the representation of binary trees.
    ## used in class Binary_search_tree's __repr__
    ## Written by a former student in the course - thanks to Amitai Cohen

    def printree(self, t, bykey=True):
        """Print a textual representation of t
        bykey=True: show keys instead of values"""
        # for row in trepr(t, bykey):
        #        print(row)
        return self.trepr(t, bykey)

    def trepr(self, t, bykey=False):
        """Return a list of textual representations of the levels in t
        bykey=True: show keys instead of values"""
        if t == None:
            return ["#"]
        if t.getParent() is not None:
            st = str(t.value) + " " + str(t.size) + " " + str(t.height) + " " + str(t.bf) + " " + str(
                t.parent.value)
        else:
            st = str(t.value) + " " + str(t.size) + " " + str(t.height) + " " + str(t.bf) + " " + str(t.parent)
        thistr = st if bykey else str(t.val)

        return self.conc(self.trepr(t.left, bykey), thistr, self.trepr(t.right, bykey))

    def conc(self, left, root, right):
        """Return a concatenation of textual represantations of
        a root node, its left node, and its right node
        root is a string, and left and right are lists of strings"""

        lwid = len(left[-1])
        rwid = len(right[-1])
        rootwid = len(root)

        result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

        ls = self.leftspace(left[0])
        rs = self.rightspace(right[0])
        result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

        for i in range(max(len(left), len(right))):
            row = ""
            if i < len(left):
                row += left[i]
            else:
                row += lwid * " "

            row += (rootwid + 2) * " "

            if i < len(right):
                row += right[i]
            else:
                row += rwid * " "

            result.append(row)

        return result

    def leftspace(self, row):
        """helper for conc"""
        # row is the first row of a left node
        # returns the index of where the second whitespace starts
        i = len(row) - 1
        while row[i] == " ":
            i -= 1
        return i + 1

    def rightspace(self, row):
        """helper for conc"""
        # row is the first row of a right node
        # returns the index of where the first whitespace ends
        i = 0
        while row[i] == " ":
            i += 1
        return i

    def append(self, val):
        return self.insert(self.length(), val)

    def getTreeHeight(self):
        return self.getRoot().getHeight()


#####################################################################

"""builds and AVLTree from a list in-order

@type lst: list
@type left: int
@type right: int
@param lst: the list containing the values from which the AVLTree will be built
@param left: a pointer
@param right: a pointer
@rtype: AVLNode
@returns: the root of the AVLTree
"""


def buildTreeFromList(lst, left, right):
    # if right < left then we return a virtual node
    if right < left:
        return AVLNode(None)

    # create a node from the center of the list so that the tree will be most balanced
    mid = (left + right) // 2
    node = AVLNode(lst[mid])

    # build left subtree of node
    node.setLeft(buildTreeFromList(lst, left, mid - 1))

    # build right subtree of node
    node.setRight(buildTreeFromList(lst, mid + 1, right))

    # update parameters of node
    node.setHeight(node.computeHeight())
    node.setSize(node.computeSize())
    node.setBF(node.computeBF())

    # setting node as parent for left and right children
    if node.getRight().isRealNode():
        node.getRight().setParent(node)
    if node.getLeft().isRealNode():
        node.getLeft().setParent(node)
    return node


"""finds the right most node in the subtree

@type node: AVLNode
@param node: a node
@rtype: AVLNode
@returns: the right most node (that is not virtual) in the subtree of node
"""


def max_node(node):
    while node.getRight().isRealNode():
        node = node.getRight()
    return node


"""finds the left most node in the subtree

@type node: AVLNode
@param node: a node
@rtype: AVLNode
@returns: the left most node (that is not virtual) in the subtree of node
"""


def min_node(node):
    while node.getLeft().isRealNode():
        node = node.getLeft()
    return node


"""finds the successor node of x in the tree

@type x: AVLNode
@param x: a node
@rtype: AVLNode
@returns: the successor of x
"""


def successor(x):
    # the node has a right subtree and therefore the successor of x is the min node in the left subtree
    if x.getRight().isRealNode():
        return min_node(x.getRight())

    # find the lowest ancestor y such that x is in the left subtree of y
    y = x.getParent()
    while y.isRealNode() and x == y.getRight():
        x = y
        y = x.getParent()
    return y


"""merges two lists into a sorted list

@type lst1: list
@type lst2: list
@param lst1: a list
@param lst2: a list
@rtype: list
@returns: a sorted list
"""


def merge(lst1, lst2):
    # if the first list is empty then nothing needs to be merged. return second list
    if len(lst1) == 0:
        return lst2

    # if the second list is empty then nothing needs to be merged. return first list
    if len(lst2) == 0:
        return lst1

    result = []
    index1 = index2 = 0

    # go through both lists until all the elements make it into the result list in sorted order
    while len(result) < len(lst1) + len(lst2):
        # the current element in the first list is smaller/equal to the current element in the second list
        if lst1[index1] <= lst2[index2]:
            result.append(lst1[index1])
            index1 += 1

        # the current element in the second list is larger than the current element in the first list
        else:
            result.append(lst2[index2])
            index2 += 1

        # if we have reached the end of one list

        # add the remaining elements from the first list
        if index2 == len(lst2):
            result += lst1[index1:]
            break

        # add the remaining elements from the second list
        if index1 == len(lst1):
            result += lst2[index2:]
            break

    return result


"""sorts a list

@type lst: list
@param lst: a list
@rtype: list
@returns: a sorted copy of lst
"""


def merge_sort(lst):
    # if the input list contains fewer than two elements then nothing needs to be sorted
    if len(lst) < 2:
        return lst

    mid = len(lst) // 2

    # sort the list by recursively splitting the input into two equal halves, sorting each half, and merging them together
    return merge(merge_sort(lst[:mid]), merge_sort(lst[mid:]))


"""shuffles a list in place

@type lst: list
@param lst: a list
"""
def shuffle(lst):
    for index in range(len(lst) - 1, 0, -1):
        # choose a random element in the list to swap places with the current element
        other = random.randint(0, len(lst) - 1)
        if other == index:
            continue
        lst[index], lst[other] = lst[other], lst[index]



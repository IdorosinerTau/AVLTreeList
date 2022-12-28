# username - idorosiner
# id1      - 209617000
# name1    - Ido Rosiner
# id2      - 206627820
# name2    - Tomer Rudnitzky

import random

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

	"""returns the left child - O(1) time complexity
	
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""

	def getLeft(self):
		return self.left

	"""returns the right child - O(1) time complexity

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""

	def getRight(self):
		return self.right

	"""returns the parent - O(1) time complexity

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

	def getParent(self):
		return self.parent

	"""return the value - O(1) time complexity

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""

	def getValue(self):
		return self.value

	"""returns the height - O(1) time complexity

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

	def getHeight(self):
		return self.height

	"""returns the size - O(1) time complexity

	@rtype: int
	@returns: the size of self, 0 if node is virtual
	"""

	def getSize(self):
		return self.size

	"""returns the balance factor - O(1) time complexity

	@rtype: int
	@returns: the balance factor of self
	"""

	def getBF(self):
		return self.bf

	"""sets left child - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setLeft(self, node):
		self.left = node

	"""sets right child - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setRight(self, node):
		self.right = node

	"""sets parent - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setParent(self, node):
		self.parent = node

	"""sets value - O(1) time complexity

	@type value: str
	@param value: data
	"""

	def setValue(self, value):
		self.value = value

	"""sets the height - O(1) time complexity

	@type h: int
	@param h: data
	"""

	def setHeight(self, h):
		self.height = h

	"""sets the size - O(1) time complexity

	@type s: int
	@param s: data
	"""

	def setSize(self, s):
		self.size = s

	"""sets the balance factor - O(1) time complexity

	@type bf: int
	@param bf: data
	"""

	def setBF(self, bf):
		self.bf = bf

	"""returns whether self is not a virtual node  - O(1) time complexity

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

	def isRealNode(self):
		if self.height == -1:
			return False
		return True

	"""computes the balance factor - O(1) time complexity

	@rtype: int
	@returns: the balance factor that self should have, i.e height(left_subtree) - height(right_subtree)
	"""

	def computeBF(self):
		return (1 + self.getLeft().getHeight()) - (1 + self.getRight().getHeight())

	"""computes the height - O(1) time complexity

	@rtype: int
	@returns: the height that self should have, i.e 1 + max(height(left_subtree), height(right_subtree))
	"""

	def computeHeight(self):
		return 1 + max(self.getLeft().getHeight(), self.getRight().getHeight())

	"""computes the size - O(1) time complexity

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
		if self.empty():
			return True
		return self.getRoot().recc()

	"""returns the root of the tree representing the list - O(1) time complexity

	@rtype: AVLNode
	@returns: the root, virtual node if the list is empty
	"""

	def getRoot(self):
		return self.root

	"""returns the maximum node of the tree representing the list - O(1) time complexity

	@rtype: AVLNode
	@returns: the right most node, None if the list is empty
	"""

	def getMax(self):
		return self.max

	"""returns the minimum node of the tree representing the list - O(1) time complexity

	@rtype: AVLNode
	@returns: the left most node, None if the list is empty
	"""

	def getMin(self):
		return self.min

	"""sets the size of the tree representing the list - O(1) time complexity

	@type s: int
	@param s: data
	"""

	def setSize(self, s):
		self.size = s

	"""sets the root of the tree representing the list - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setRoot(self, node):
		self.root = node

	"""sets the max of the tree representing the list - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setMax(self, node):
		self.max = node

	"""sets the min of the tree representing the list - O(1) time complexity

	@type node: AVLNode
	@param node: a node
	"""

	def setMin(self, node):
		self.min = node

	"""returns whether the list is empty - O(1) time complexity

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""

	def empty(self):
		if not self.getRoot().isRealNode():
			return True
		return False

	"""retrieves the value of the i'th item in the list - O(log(i)) time complexity

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the value of the i'th item in the list
	"""

	def retrieve(self, i):
		if i >= self.length() or i < 0:
			return None
		return self.treeSelect(i + 1).getValue()

	"""inserts val at position i in the list - O(log(n)) time complexity

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def insert(self, i, val):
		# the user is trying to insert an element that's out of range
		if i > self.length() or i < 0:
			return -1

		# initializing a node to insert into the tree
		node_to_insert = AVLNode(val)
		node_to_insert.setRight(AVLNode(None))
		node_to_insert.setLeft(AVLNode(None))
		node_to_insert.getRight().setParent(node_to_insert)
		node_to_insert.getLeft().setParent(node_to_insert)
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

	"""deletes the i'th item in the list - O(log(n)) time complexity

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

	def delete(self, i):
		#there are no elements in the tree to delete
		if self.empty():
			return -1

		# the user is trying to delete an element that's out of range
		if i >= self.length() or i < 0:
			return -1

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

		# preparing deletion from the start of the tree representing a list
		elif i == 0:
			node_to_del = self.getMin()
			self.setMin(self.successor(node_to_del))

		# preparing deletion from the center of the tree representing a list
		else:
			node_to_del = self.treeSelect(i + 1)

		# deleting the node from the tree
		node_to_rebalance = self.delete_node(node_to_del)

		# rabalancing the tree
		num_rebalancing = self.rebalanceTree(node_to_rebalance)
		self.setSize(self.length() - 1)
		self.setMax(self.treeSelect(self.length()))
		return num_rebalancing

	"""deletes a certain node from the tree representing the list - O(1) time complexity

	@type node_to_del: AVLNode
	@param node_to_del: the node that will be deleted from the tree 
	@pre: node_to_del in self
	@rtype: AVLNode
	@returns: the node from which we need to start rebalancing
	"""

	def delete_node(self, node_to_del):
		# deleting node with no child
		if not node_to_del.getLeft().isRealNode() and not node_to_del.getRight().isRealNode():
			parent = node_to_del.getParent()
			if parent.getLeft() is node_to_del:
				parent.setLeft(node_to_del.getLeft())
			else:
				parent.setRight(node_to_del.getLeft())
			node_to_del.getLeft().setParent(parent)
			return parent

		# deleting node with only right child
		elif not node_to_del.getLeft().isRealNode():
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

		# node with two children:
		else:
			# get the inorder successor and copy it's value to this node
			suc_node = self.successor(node_to_del)
			node_to_del.setValue(suc_node.getValue())

			# delete the inorder successor
			return self.delete_node(suc_node)

	"""returns the value of the first item in the list - O(1) time complexity

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""

	def first(self):
		if not self.getRoot().isRealNode():
			return None
		return self.getMin().getValue()

	"""returns the value of the last item in the list - O(1) time complexity

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""

	def last(self):
		if not self.getRoot().isRealNode():
			return None
		return self.getMax().getValue()

	"""returns an array representing list - O(n) time complexity

	@rtype: list
	@returns: a list of strings representing the data structure
	"""

	def listToArray(self):
		L = []
		self.inorder(self.getRoot(), L)
		return L

	"""appends the nodes of the tree in-order to the list - O(n) time complexity

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

	"""returns the size of the list - O(1) time complexity

	@rtype: int
	@returns: the size of the list
	"""

	def length(self):
		return self.getRoot().getSize()

	"""sort the info values of the list - O(nlog(n)) time complexity

	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	"""

	def sort(self):
		# get the list that represents the tree
		tree_arr = self.listToArray()

		# self is empty
		if not tree_arr:
			return AVLTreeList()

		# create list of all None elements of tree_arr
		none_lst = [None for val in tree_arr if val is None]

		# create list of all the elements that are not None in tree_arr
		tree_arr = [val for val in tree_arr if val is not None]

		# create a sorted copy of the list where all None elements are at the start
		org_arr = none_lst + self.merge_sort(tree_arr)

		# create new AVL tree
		new_tree = AVLTreeList()
		# build a tree from sorted list and set it's root to be new_tree's root
		new_tree.setRoot(self.buildTreeFromList(org_arr, 0, len(org_arr) - 1))
		new_tree.getRoot().setParent(AVLNode(None))
		# set new_tree's size
		new_tree.setSize(new_tree.length())
		#set new_tree's min pointer
		new_tree.setMin(self.min_node(new_tree.getRoot()))
		# set new_tree's max pointer
		new_tree.setMax(self.max_node(new_tree.getRoot()))
		return new_tree

	"""permute the info values of the list - O(n) time complexity

	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""

	def permutation(self):
		# get the list that represents the tree
		tree_arr = self.listToArray()

		# self is empty
		if not tree_arr:
			return AVLTreeList()

		# shuffle the list in place
		self.shuffle(tree_arr)

		# create new AVL tree
		new_tree = AVLTreeList()
		# build a tree from shuffled list and set it's root to be new_tree's root
		new_tree.root = self.buildTreeFromList(tree_arr, 0, len(tree_arr) - 1)
		new_tree.getRoot().setParent(AVLNode(None))
		# set new_tree's size
		new_tree.setSize(new_tree.length())
		# set new_tree's min pointer
		new_tree.setMin(self.min_node(new_tree.getRoot()))
		# set new_tree's max pointer
		new_tree.setMax(self.max_node(new_tree.getRoot()))
		return new_tree

	"""concatenates lst to self - O(log(n))

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

	def concat(self, lst):
		#absolute difference in height between both trees before any changes
		height_diff = abs(self.getRoot().getHeight() - lst.getRoot().getHeight())
		new_size = self.length() + lst.length()

		#both self and lst are empty
		if self.empty() and lst.empty():
			return 0

		#lst is empty
		if lst.empty():
			return self.getRoot().getHeight()

		#self is empty
		if self.empty():
			self.setRoot(lst.getRoot())
			self.setMin(lst.getMin())
			self.setMax(lst.getMax())
			self.setSize(new_size)
			return lst.getRoot().getHeight()

		#self has only one element
		if self.length() == 1:
			lst.insert(0, self.getRoot().getValue())
			self.setRoot(lst.getRoot())
			self.setMin(lst.getMin())
			self.setMax(lst.getMax())
			self.setSize(new_size)
			return height_diff

		#lst has only one element
		if lst.length() == 1:
			self.insert(self.length(), lst.getRoot().getValue())
			return height_diff

		#delete the max node of self and use it as a connector to both trees like we learnt in class
		x = self.getMax()
		self.delete(self.length() - 1)

		#tree heights after deletion
		lst_height = lst.getRoot().getHeight()
		self_height = self.getRoot().getHeight()

		#both trees have the same height
		if self_height == lst_height:
			self.joinSelfSame(x, lst)

		#the height of self is bigger
		elif self_height > lst_height:
			self.joinSelfBigger(x, lst)
			#rebalance tree after connecting
			self.rebalanceTree(x)

		#the height of self is smaller
		else:
			self.joinSelfSmaller(x, lst)
			# rebalance tree after connecting
			self.rebalanceTree(x)
		self.setMax(lst.getMax())
		self.setSize(new_size)
		return height_diff

	"""connects lst to self if height of self is bigger than height of lst - O(log(n))

	@type x: AVLNode
	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@param x: node that is used as a connector to both trees like we learnt in class
	"""
	def joinSelfBigger(self, x, lst):
		#height of lst
		connect_height = lst.getRoot().getHeight()

		# search for a node in self with the same height or 1 less than height of lst
		node_self = self.getRoot()
		while node_self.getHeight() > connect_height:
			node_self = node_self.getRight()

		#if the node is the root of self then it's like connecting trees with the same height
		if self.getRoot() is node_self:
			return self.joinSelfSame(x, lst)

		#connect x to both trees and both trees to x
		x.setLeft(node_self)
		x.setRight(lst.getRoot())
		x.setParent(node_self.getParent())
		node_self.getParent().setRight(x)
		lst.getRoot().setParent(x)
		node_self.setParent(x)

	"""connects lst to self if height of self is smaller than height of lst - O(log(n))

	@type x: AVLNode
	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@param x: node that is used as a connector to both trees like we learnt in class
	"""
	def joinSelfSmaller(self, x, lst):
		# height of self
		connect_height = self.getRoot().getHeight()

		# search for a node in lst with the same height or 1 less than height of self
		node_lst = lst.getRoot()
		while node_lst.getLeft().isRealNode() and node_lst.getHeight() > connect_height:
			node_lst = node_lst.getLeft()

		# if the node is the root of lst then it's like connecting trees with the same height
		if lst.getRoot() is node_lst:
			return self.joinSelfSame(x, lst)

		# connect x to both trees and both trees to x
		x.setRight(node_lst)
		x.setLeft(self.getRoot())
		x.setParent(node_lst.getParent())
		node_lst.getParent().setLeft(x)
		self.getRoot().setParent(x)
		node_lst.setParent(x)

		#self is a subtree of lst and therefore the new root is the root of lst
		self.setRoot(lst.getRoot())

	"""connects lst to self if self and lst have the same height - O(1)

	@type x: AVLNode
	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@param x: node that is used as a connector to both trees like we learnt in class
	"""
	def joinSelfSame(self, x, lst):
		# connect x to the root of both trees and connect the root of both trees to x
		x.setLeft(self.getRoot())
		x.setRight(lst.getRoot())
		x.setParent(AVLNode(None))
		self.getRoot().setParent(x)
		lst.getRoot().setParent(x)

		#compute and set all of x's new parameters
		x.setHeight(x.computeHeight())
		x.setSize(x.computeSize())
		x.setBF(x.computeBF())

		#x is now the new root of the whole tree
		self.setRoot(x)


	"""searches for a *value* in the list - O(n) time complexity

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""

	def search(self, val):
		#create a list representing the tree
		arr = self.listToArray()

		#find the index of val in the list
		for i, value in enumerate(arr):
			if val == value:
				return i

		#val is not in the list
		return -1

	"""performs a right rotation on the node - O(1) time complexity

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
		if self.getRoot() is not node:
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

	"""performs a left rotation on the node - O(1) time complexity

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
		if self.getRoot() is not node:
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

	"""rebalances the tree representing the list - O(log(n))

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

			# if node's balance factor is illegal
			if abs(node.getBF()) >= 2:
				# right subtree is too big
				if node.getBF() <= -2:
					#balance tree like we learnt in class
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
					# balance tree like we learnt in class
					left_son = node.getLeft()
					if left_son.getBF() == -1:
						AVLTreeList.rotateLeft(self, node.getLeft())
						AVLTreeList.rotateRight(self, node)
						num_rebalancing += 2
					else:
						AVLTreeList.rotateRight(self, node)
						num_rebalancing += 1
			node = node.getParent()
		return num_rebalancing

	"""finds and returns the node in index i - 1 of the list - O(log(i))

	@type i: int
	@pre: 0 < i <= self.length()
	@param i: the i'th smallest element in the tree
	@rtype: AVLNode
	@returns: the i'th node of the tree representing the list
	"""

	def treeSelect(self, i):
		node = self.getMin()
		# locate the node that has at least i-1 children
		while node.getSize() < i:
			node = node.getParent()
		return self.treeSelectHelper(node, i)

	"""a helper function to treeSelect - O(log(i)) time complexity

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

	"""finds the right most node in the subtree - O(log(n)) time complexity

	@type node: AVLNode
	@param node: a node
	@rtype: AVLNode
	@returns: the right most node (that is not virtual) in the subtree of node
	"""

	def max_node(self, node):
		while node.getRight().isRealNode():
			node = node.getRight()
		return node

	"""finds the left most node in the subtree - O(log(n)) time complexity

	@type node: AVLNode
	@param node: a node
	@rtype: AVLNode
	@returns: the left most node (that is not virtual) in the subtree of node
	"""

	def min_node(self, node):
		while node.getLeft().isRealNode():
			node = node.getLeft()
		return node

	"""finds the predecessor node of x in the tree - O(log(n)) time complexity

	@type x: AVLNode
	@param x: a node
	@rtype: AVLNode
	@returns: the predecessor of x
	"""

	def predecessor(self, x):
		# the node has a left subtree and therefore the predecessor of x is the max node in the left subtree
		if x.getLeft().isRealNode():
			return self.max_node(x.getLeft())

		# find the lowest ancestor y such that x is in the right subtree of y
		y = x.getParent()
		while y.isRealNode() and x == y.getLeft():
			x = y
			y = x.getParent()
		return y

	"""finds the successor node of x in the tree - O(log(n)) time complexity

	@type x: AVLNode
	@param x: a node
	@rtype: AVLNode
	@returns: the successor of x
	"""

	def successor(self, x):
		# the node has a right subtree and therefore the successor of x is the min node in the left subtree
		if x.getRight().isRealNode():
			return self.min_node(x.getRight())

		# find the lowest ancestor y such that x is in the left subtree of y
		y = x.getParent()
		while y.isRealNode() and x == y.getRight():
			x = y
			y = x.getParent()
		return y

	"""builds and AVLTree from a list in-order - O(n) time complexity

	@type lst: list
	@type left: int
	@type right: int
	@param lst: the list containing the values from which the AVLTree will be built
	@param left: a pointer
	@param right: a pointer
	@rtype: AVLNode
	@returns: the root of the AVLTree
	"""

	def buildTreeFromList(self, lst, left, right):
		# if right < left then we return a virtual node
		if right < left:
			return AVLNode(None)

		# create a node from the center of the list so that the tree will be most balanced
		mid = (left + right) // 2
		node = AVLNode(lst[mid])

		# build left subtree of node
		node.setLeft(self.buildTreeFromList(lst, left, mid - 1))

		# build right subtree of node
		node.setRight(self.buildTreeFromList(lst, mid + 1, right))

		# update parameters of node
		node.setHeight(node.computeHeight())
		node.setSize(node.computeSize())
		node.setBF(node.computeBF())

		# setting node as parent for left and right children
		node.getRight().setParent(node)
		node.getLeft().setParent(node)
		return node

	"""merges two lists into a sorted list - O(n + m) time complexity

	@type lst1: list
	@type lst2: list
	@param lst1: a list
	@param lst2: a list
	@rtype: list
	@returns: a sorted list
	"""

	def merge(self, lst1, lst2):
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

	"""sorts a list - O(nlog(n)) time complexity

	@type lst: list
	@param lst: a list
	@rtype: list
	@returns: a sorted copy of lst
	"""

	def merge_sort(self ,lst):
		# if the input list contains fewer than two elements then nothing needs to be sorted
		if len(lst) < 2:
			return lst

		mid = len(lst) // 2

		# sort the list by recursively splitting the input into two equal halves, sorting each half, and merging them together
		return self.merge(self.merge_sort(lst[:mid]), self.merge_sort(lst[mid:]))

	"""shuffles a list in place - O(n) time complexity

	@type lst: list
	@param lst: a list
	"""

	def shuffle(self, lst):
		for index in range(len(lst) - 1, 0, -1):
			# choose a random element in the list to swap places with the current element
			other = random.randint(0, len(lst) - 1)
			if other == index:
				continue
			lst[index], lst[other] = lst[other], lst[index]





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





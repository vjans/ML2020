import numpy as np
import scipy as sp
import os
import scipy.io.arff as arff

import sys
import pandas

# global_variable
attributes_full = {}

#Task 1
def parser(path):
	"""
	function which parses the data from an arff file 
	
	@param path: string containig the path to file	
	
	@return array containing the data	
	@raise FileNotFoundError exception in case if the path does not point to a valid file
	"""
		
	start = 0
	data = []
	attributes = []
	counter = 0
	if os.path.isfile(path):
		in1 = open(path, "r")
		for x in in1:
			if x.startswith('@attribute'):
				tmp = x.replace("{","").replace("}","").replace("\n","").replace(",","").split(" ")
				values = tmp[2:]
				attributes.append(tmp[1])
				attributes_full.update({tmp[1]:values})
			if start:
				data.append(x.replace("\n","").split(","))
			if x.startswith('@data'):
				start=1
		return pandas.DataFrame(np.array(data), columns = attributes, index = range(0, len(data)) )
	else:	
		sys.exit("FileNotFoundError")
	
	

#Task 3
def entropyOnSubset(data, class_label, indices=None):
	"""
	function to calculate the entropy of the given dataset and labels
	
	@param data: array containing the data 
	@param class_label: class label of instances
	@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None

	@return: the entropy as a float
	"""
	subset = []
	if indices == None:
		subset = data[:]
	else: 
		subset = data.loc[indices]
	
	subset = subset[class_label].tolist()
	values = list(set(subset))
	entropy = 0
	
	for i in values:
		pV = subset.count(i)/len(subset)
		entropy -= pV * np.log2(pV)
	
	return entropy


#Task 4
def informationGain(data, class_label, attribute, indices=None):
	"""
	function which returns the information gain using a given dataset, indices, and attributes

	@param data: array containing the data
	@param class_label: class label of instances
	@param attribute: selected attribute
	@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None
			
	@return: informationGain as a float.
	"""
	subset = []
	if indices == None:
		subset = data[:]
	else: 
		subset = data.loc[indices]
	
	sublist = subset[attribute].tolist()
	values = list(set(sublist))
	infoGain = entropyOnSubset(subset, class_label)
	
	#print (sublist)
	
	for i in values:
		index = list(subset.index[subset[attribute] == i])
		infoGain -= sublist.count(i)/len(sublist) * entropyOnSubset(subset, class_label, index)

	
	return infoGain


#Task 5
def attributeSelection(data, attributes, class_label, indices=None):
	"""
	function which selects a the best attribute for the current node given an array of possible attributes and the data with their labels
	
	@param data: array containing the data
	@param attributes: list of all attributes from which the optmial one should be selected
	@param class_label: class label of instances
	@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None
		
	@return: index of the best attribute
	"""
	best = 0
	bestIndex = 0
	counter = 0
	for i in attributes:
		infoG = informationGain(data, class_label, i, indices)
		if infoG > best:
			best = infoG
			bestIndex = counter
		counter += 1 
	
	return bestIndex


#Task 2
class Node:
	"""
	contain the structure of a decision tree: it has either subnodes associated with the corresponding attribute values or is a leaf node. 
	To set the arributes or leaf value use functions, do not access the parameters directly!
	"""
	def __init__(self, name="root", children=None):
		self.name = name
		self.children = []
		if children is not None:
			for child in children:
				self.add_child(child)
		
	def add_child(self, child):
		self.children.append(child)
	
	def get_name(self):
		return self.name()
	
	def trainNode(self, data, attributes, class_label, indices=None):
		"""
		ID3 based algorithm to find the optimal attribtue
			
		@param data: array containing the data
		@param attributes: list of all attributes from which the optmial one should be selected
		@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None
		"""
		optimalAttribute = attributeSelection(data, attributes, class_label, indices)	
		
		return optimalAttribute
		   
			
	
	def print_recursive(self, indents):
		"""
		function to print the entire tree
		
		@param indents: Integer value of how many indents there should be when printing this node information
		"""
		
		
		ind = "\t"
		output = indents * ind + self.name
		print(output)
		for i in self.children:
			i.print_recursive(indents+1)
			
	def get_bebe(self, label):
		#print("Name: " + self.name)
		#print(label)
		#print(len(self.children))
		clone = label[:]
		if len(self.children) == 0:
			return str(self.name)
		for j in clone:
			for i in self.children:
				if i.name == j:
					clone.remove(j)
					return i.get_bebe(clone)
				
	
class DecisionTree:
	"""
	class which represents the decision tree and holds the reference to root node
	"""
	def __init__(self):
		self.root = Node()
			  
		
	def trainModel(self, data, attributes, class_label):
		"""
		function to train the model using a given dataset
		
		@param data: array containing the data with their labels
		@param attributes: list of all attributes from which the optmial one should be selected
		"""
		
		self.trainModelOnSubset(data, attributes, class_label)
		
		
		
	def trainModelOnSubset(self, data, attributes, class_label, indices=None, startNode = None):
		"""
		train a certain part of the tree starting with the given startNode based on a subset of the data indicated by the indices array
		
		@param data: array containing the data with their labels
		@param attributes: list of all attributes from which the optmial one should be selected
		@param class_label: list of class values 
		@param indices: list of indices of the chosen subset from the given complete dataset
			If set to None the entire dataset is used. Default value: indices = None
		@param startNode: the root node of the subtree. By default the start node is the root of the tree. Default value: startNode = None
		"""
		
		#print (data)
		#print("_________---_________")
		
		if startNode == None:
			startNode = self.root
		
		#print(startNode.name)
		
		#print("Attributes: " + str(attributes))
		
		new_attributes=attributes[:]
		
		if indices == None:
			subset = data			
		else:
			subset = data.loc[indices]
		
		
		if (len(new_attributes) > 0 and not subset.empty):
			bestAttribute_index = startNode.trainNode(data, new_attributes, class_label, indices)
			bestAttribute = new_attributes[bestAttribute_index]
			new_attributes.pop(bestAttribute_index)
			values = attributes_full[bestAttribute]
			
			
			
			
			for i in values:
				node = Node(i)
				startNode.add_child(node)
				#print(node.get_name())
				#print(bestAttribute)
				index = list(subset.index[subset[bestAttribute] == i])
				self.trainModelOnSubset(subset, new_attributes, class_label, index, node)
			
		else:
			bestAttribute=class_label
			values = attributes_full[bestAttribute]
			sublist = data[bestAttribute].tolist()
			top = 0
			name = "test"
			for i in values:
				if sublist.count(i) > top:
					name = i
					top = sublist.count(i)
			node = Node(name)
			startNode.add_child(node)

	"""
	function which returns the expected class value for the given dataset
	"""					
		
		
			
	def print_tree(self):
		"""
		function to print the entire tree
		"""
		self.root.print_recursive(0)
		
	def get_prediction(self, data, class_label):
		"""
		function which predicts the class label for the given data
		
		@param data: list of test data
		"""
		accuracy = 0
		hit=0
		count=0
		for index, row in test.iterrows():
			count += 1
			tmp = self.get_classLabel(row.tolist(),row[class_label])
			#print (tmp)
			if tmp:
				hit+=1
		#print ("hit "+ str(hit) )
		accuracy = hit/count
		
		return accuracy
	
	def get_classLabel(self, dataset, class_label):
		"""
		function which returns the expected class value for the given dataset
		"""     
		node = self.root
		broken=0
		
		#print("BEBE:" + str(node.get_bebe( dataset)))
		
		if (node.get_bebe( dataset) == class_label ):
			return 1
		else:
			return 0

				
			
#Task 6: test on dataset
path = 'weather.nominal.arff'
data_arff = parser(path)
class_label = "play"


print("dataset:")
print(data_arff)
print()

indices = [ 1,3,4,5 ]
print("Entropy on subset: " + str(entropyOnSubset(data_arff, class_label, indices)))
print("Entropy on full set: " + str(entropyOnSubset(data_arff, class_label)))

print("Information Gain on humidity: " + str(informationGain(data_arff, class_label, "humidity")))

attributes = [ "temperature", "outlook", "humidity", "windy" ] 
print("Best index: " + str(attributeSelection(data_arff, attributes, class_label)))

print(attributes_full)
print()

print("Tree: \n")
ds = DecisionTree()
ds.trainModel(data_arff, attributes, class_label)
ds.print_tree()



test = data_arff.loc[indices]
print ("test_set:")
print (test)
print("Accuracy: " + str(ds.get_prediction(test,class_label)))





#ToDo: Create train and test data and get resulted tree



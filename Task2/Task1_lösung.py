import numpy as np
import scipy as sp
import os
import scipy.io.arff as arff

#Task 1
def parser(path):
    """
    function which parses the data from an arff file 
    
    @param path: string containig the path to file    
    
    @return array containing the data    
    @raise FileNotFoundError exception in case if the path does not point to a valid file
    """
    #Declaratives as constant to avoid misspelling in code
    RELATION = 'relation'
    ATTRIBUTE = 'attribute'
    DATA = 'data'
    
    #Create dictionary holding the arff information
    data = {RELATION : [],
            ATTRIBUTE : [],
            DATA : []}
    
    #Read the file and analyse the data
    with open(path) as file:
        for line in file.readlines():            
            #Check if line is empty
            if line.strip() == '':
                continue  
                
            #Check if line contains the relation
            elif '@'+RELATION in line:
                data[RELATION].append(line.replace('@'+RELATION,'').strip())
                
            #Check if line contains an attribute
            elif '@'+ATTRIBUTE in line:
                #Format the line
                line = line.replace('@'+ATTRIBUTE,'')
                line = line.replace('{', '')
                line = line.replace(',', '')
                line = line.replace('}', '')
                line = line.strip()
                line = line.split(' ')
                #Append the data to the attributes array
                data[ATTRIBUTE].append({'name' : line[0], 'values' : line[1:]})
                
            #If the line is not one of the others, it has to be data
            elif '@'+DATA not in line:
                line = line.split(',')
                #strip each element of the line
                for i in range(len(line)):
                    line[i] = line[i].strip()
                #Add data to dictionary
                data[DATA].append(line)
    return data
	
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
    #Create subset of dataset if array with indices is given
    if indices is not None:
        data = np.array(data)[indices]
    #Count how often every label occours in the subset
    class_label_counts = np.zeros(len(class_label['values']))
    for i in range(len(data)):
        for j in range(len(class_label['values'])):
            if data[i][class_label['name']] == class_label['values'][j]:
                class_label_counts[j] += 1
                
    #Always check if one of the elements of class_label_counts is 0 otherwise one obtains nan due to log2
    #but x log(x) = 0 for x -> 0. This messes up the result!
    entropy = -np.sum(class_label_counts[class_label_counts!=0] / len(data) * np.log2(class_label_counts[class_label_counts!=0]/len(data)))
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
    #Create subset of dataset if array with indices is given
    if indices is not None:
        data = np.array(data)[indices]
    #Cacluate the entropy using the attributes currently stored by the node
    infoGain = entropyOnSubset(data, class_label)
    #Calculate the entropy of every new attribute
    for val in attribute['values']:
        #Create index list for slicing
        indices = np.array([True if data[i][attribute['name']] == val else False for i in range(len(data))])
        #Calculate correction of entropy
        infoGain -= len(indices[indices])/len(indices) * entropyOnSubset(data, class_label, indices=indices)
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
    #Create subset of the data if indices are given
    if indices is not None:
        data = data[indices]
    
    #Check all attributes. Start with negative attr_best_val since the information gain is the intverall [0,1]
    bestIndex = None
    attribute_best_value = -1
    for i in range(len(attributes)):
        attribute = attributes[i]
        #Calculate the information gain
        infoGain = informationGain(data, class_label, attribute)
        #Check if the new information gain is better than the previous one
        if attribute_best_value < infoGain:
            attribute_best_value = infoGain
            bestIndex = i
    
    return bestIndex



#Internal function which sets this node to leaf node using the most common class label
def get_mostCommonClassLabel(data, attribute, indices=None):
    """
    This function calculates the most common attribute of the given data array
    
    @param data: Array containing the data sets given as dictionaries
    @param attribute: Attribute of which the most common value has to be calculated
    @param indices: Array containing the indices which determine the subset of the given dataset. 
            If set to None the entire dataset is used. Default value: indices == None
    """
    #Slice data if index array is given
    if indices is not None:
        data = data[indices]

    #Count the occourences
    counts = np.zeros(len(attribute['values']))
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i][attribute['name']] == attribute['values']:
                counts += 1         
    #Return the value of the class label which occoures the most often
    return attribute['values'][np.argmax(counts)]


#Task 2
class Node:
    """
    contain the structure of a decision tree: it has either subnodes associated with the corresponding attribute values or is a leaf node. 
    To set the arributes or leaf value use functions, do not access the parameters directly!
    """
    def __init__(self):
        #Parameters for nodes
        self.attribute_name = None
        self.descendants  = {}
        
        #Parameters for leafs
        self.leaf_value = None
        
        
    def trainNode(self, data, attributes, class_label, indices=None):
        """
        ID3 based algorithm to find the optimal attribtue
            
        @param data: array containing the data
        @param attributes: list of all attributes from which the optmial one should be selected
        @param indices: list of indices of the chosen subset from the given complete dataset
            If set to None the entire dataset is used. Default value: indices = None
        """
        #Create subset of the data if indices are given
        if indices is not None:
            data = data[indices]
            
        #Check if data is pure
        data_classLabels = []
        for i in range(len(data)):
            data_classLabels.append(data[i][class_label['name']])
        data_classLabels_unique = np.unique(np.array(data_classLabels))
        if len(data_classLabels_unique) == 1:
            #If data is pure, set node to leaf node
            self.set_leafNode(data_classLabels_unique[0])
            
        #If no attributes are given on which to optimize, get most common class label of data
        elif len(attributes) == 0:
            self.set_leafNode(get_mostCommonClassLabel(data, class_label))
            
        else:
            #Get index of best attribute
            attribute_index = attributeSelection(data, attributes, class_label)
            attribute = attributes[attribute_index]
            #Save the attribute name
            self.attribute_name = attribute['name']
            #Create attribute array without the current optimal attribute
            attributes = np.delete(attributes, attribute_index)
            for val in attribute['values']:
                #Create subset containing all values which are equal the current one of the loop
                subset = []
                for i in range(len(data)):
                    if data[i][attribute['name']] == val:
                        subset.append(data[i])
                
                #If subset is empty set node to leaf node with common class label as leaf value
                if len(subset) == 0:
                    node = Node()
                    node.set_leafNode(get_mostCommonClassLabel(data, class_label))
                    self.descendants.update({val : node})
                    
                #If subset is not empty, add a node and run the optimization
                else:
                    node = Node()
                    node.trainNode(subset, attributes, class_label)
                    self.descendants.update({val : node})
                    
            
            
    def set_leafNode(self, leaf_value):
        """
        Function which is used to set the node to a leaf node and set its value
        
        @param leaf_value: The value this leaf node will have
        """
        #Make sure there are no subnodes
        self.descendants.clear()
        #Set leaf value
        self.leaf_value = leaf_value
        
        
    def get_nextNode(self, value):
        """
        Function which returns the next node based on the given value
        
        @param value: The value for which the next node shall be returned. If value is unkown None gets returned.
        """
        #If value is in the dictionary, return the corresponding node
        if value in self.descendants:
            return self.descendants[value]
        return None
    
    def get_leaf(self, data):
        """
        Function to use the given dataset and search the tree for the expected class value
        
        @param data: Dictionary containing the attribute name as key and the attribute value as value
        
        @return: This function returns the leaf value of the given dataset. If the data given by the dataset is unkown 
                None will be returned.
        """
        #If this is the leaf node, return the leaf value
        if self.leaf_value:
            return self.leaf_value
        #If this is not the leaf node, then check if given data is known and return the corresponding subnode, otherwise return None
        if self.attribute_name in data:
            return self.get_nextNode(data[self.attribute_name]).get_leaf(data)
        return None
         
           
            
    
    def print_recursive(self, indents):
        """
        function to print the entire tree
        
        @param indents: Integer value of how many indents there should be when printing this node information
        """
        ind = '\t'*indents
        if self.leaf_value:
            print(ind,'Leaf: ', self.leaf_value)
        else:
            print(ind, 'Node name: ', self.attribute_name)
            for key in self.descendants.keys():
                print(ind, '\tValue: ', key)
                self.descendants[key].print_recursive(indents+1)
            
    
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
        if not startNode:
            startNode = self.root
        if indices:
            data=data[indices]
        #Start the training at the root node
        startNode.trainNode(data, attributes, class_label)
        
                            
    def get_classLabel(self, dataset):
        """
        function which returns the expected class value for the given dataset
        """                    
        self.root.get_leaf(dataset)
        
            
    def print_tree(self):
        """
        function to print the entire tree
        """
        self.root.print_recursive(0)
        
    def get_prediction(self, data):
        """
        function which predicts the class label for the given data
        
        @param data: list of test data
        """
        return self.root.get_leaf(data)
		
		
#Task 6: test on dataset
path = 'weather.nominal.arff'
data_arff = parser(path)
#ToDo: Create train and test data and get resulted tree
attributes = np.array(data_arff['attribute'])
#Create data array holding dictionaries
data = []
for i in range(len(data_arff['data'])):
    data_dict = {}
    for j in range(len(attributes)):
        data_dict.update({attributes[j]['name'] : data_arff['data'][i][j]})
    data.append(data_dict)

#Create test and training data    
data_training = data[:-5]
data_testing = data[-5:]

#Get calss labels
class_label = None
for i in range(len(attributes)):
    if attributes[i]['name'] == 'play':
        class_label = attributes[i]
        attributes = np.delete(attributes, i)
    
data = np.array(data)
print('Class label\n', class_label)
print('Attributes\n',attributes)
print('Data training\n',data_training)
print('Data testing\n',data_testing)

tree = DecisionTree()
tree.trainModel(data, attributes, class_label)
tree.print_tree()

import sys
import os

#functions & classes ________________________________________________________________

class Arff:
	
	relation = ""
	data = []
	attribute = {}
	
	def __init__(self, relation, attribute, data):
		self.relation = relation
		self.attribute = attribute
		self.data = data
	
	def read_arff(path):
		start = 0
		relation = ""
		in1 = open(path, "r")
		data = []
		attribute = {}
		
		for x in in1:
			if x.startswith('@relation'):
				relation = x.split(" ")[1].replace("\n","")
			if x.startswith('@attribute'):
				tmp = x.replace("{","").replace("}","").replace("\n","").replace(",","").split(" ")
				values = tmp[2:]
				attribute.update({tmp[1]:values})
			if start:
				data.append(x.replace("\n","").split(","))
			if x.startswith('@data'):
				start=1
			
		return Arff(relation, attribute, data)
	
	def print(self):
		print("Relation: " + self.relation)
		print("Attribute: " + str(self.attribute))
		print("Data: " + str(self.data))

	

#main______________________________________________________________________________	
	
path = sys.argv[1]


test = Arff.read_arff(path)

test.print()


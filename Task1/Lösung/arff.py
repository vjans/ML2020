import sys
import os

#functions & classes ________________________________________________________________

class Arff:
	
	relation = ""
	data = []
	attribute = {}
	
	def __init__(self, relation, attribute, data):
		self.relation = if (relation == none): [] else: relation
		self.attribute = if (attribute == none): [] else: attribute
		self.data = if (data == none): [] else: data
	
	def read_arff(path):
		start = 0
		in1 = open(path, "r")
	

		
		for x in in1:
			if x.startswith('@relation'):
				self.relation = x.split(" ")[1].replace("\n","")
			if x.startswith('@attribute'):
				tmp = x.replace("{","").replace("}","").replace("\n","")
				
				#checks if whitespaces between commas in attributes occur
				if ( len(tmp.split(" ")) > 3):
					values = tmp.replace(",","").split(" ")[2:]
				else:
					values = tmp[2].split(",")
					
				self.attribute.update({tmp[1]:values})
			if start:
				self.data.append(x.replace("\n","").split(","))
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


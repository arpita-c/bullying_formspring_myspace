from pandas import DataFrame, read_csv
import xml.etree.ElementTree as ET  
import matplotlib.pyplot as plt
import pandas as pd 
import os
from os import listdir
from os.path import isfile, join
import csv


outputfile = os.path.join(os.getcwd(),"outfile.csv")
op_file = open(outputfile,'w')
op_file_writer= csv.writer(op_file, delimiter='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
header=["user_id", "username", "sex", "age", "city", "province", "country", "date", "body","isbullying"]
op_file_writer.writerow(header)


# Removes Non Ascii characters in a string
def removeNonAscii(string):
	return "".join(x for x in string if ord(x)<128)


def fileparse(filename,isBullying):
	dictionary = {}
	try:
		# Read xml file 
		tree = ET.parse(filename)  
		root = tree.getroot()
	
		# Parse all tags
		for elem in root:
	
			key = elem.attrib["id"]
			value = []
	
			for subelem in elem:
				if "id" in subelem.attrib:
					value.append(subelem.attrib["id"])
				for subsub in subelem:
					if subsub.text != None:
						value.append(removeNonAscii(subsub.text))
				if subelem.text!=None and ("\n" not in subelem.text):   
					value.append(removeNonAscii(subelem.text))
	
			# Stores data in dictionary in the following format:
			value.append(isBullying)
			dictionary[key] = value
		
		for key in dictionary.keys():
			csvList=dictionary[key]
			op_file_writer.writerow(csvList)
	
	except Exception as e:
		print str(e.args)+"::"+str(e.message)



if __name__ == '__main__':
	
	currendir = os.getcwd()
	basepath  = os.path.join(currendir,"BayzickBullyingData")
	DirPath=os.path.join(basepath,"Human Concensus")
	
	listoffiles=os.listdir(DirPath)[:1]
	print listoffiles
	
	for eachfile in listoffiles:
		#Process the excel filename
		file_number = int(filter(str.isdigit, eachfile))
		print file_number
		eachfilePath=os.path.join(DirPath,eachfile)
		df = pd.read_excel(eachfilePath)
		counter=0
		
		while(True):
			try:
				print str(df.iloc[counter,0]) +"::"+str(df.iloc[counter,1])
				xml_file_name=str(df.iloc[counter,0])

				if(xml_file_name.endswith(".0")):
					xml_file_name=xml_file_name.replace(".0",".0000")

				if(xml_file_name.endswith(".001")):
					xml_file_name=xml_file_name.replace(".001",".0001")
	
				destination_path="xmlpacket"+ str(file_number) +"/"+ xml_file_name+ ".xml"
				selected_file_path=os.path.join(basepath,destination_path)
				
				if(str(df.iloc[counter,1]) == "Y"):
					isBullying="?"	
					fileparse(selected_file_path,isBullying)

				counter+=1

			except Exception as e:
				print str(e.args)+"::"+str(e.message)
				break
from bs4 import BeautifulSoup as bs
import re

import sys
import collections
import os
import csv
import urllib2

from lxml import etree
from StringIO import StringIO

document_frequency={}
frequency_doc_count={}
docsTable=[]



def execute(cursor,sql):
	"""Run the SQL statement against the object's database connection"""
	import MySQLdb

	# Try to run the sql
	try:
		cursor.execute(sql)
		return True

	except MySQLdb.Warning, e:
		# Log the warning as an error in the error log
		print(e.message)
		# Log this warning in the Watson database
		print("Database Warning", e.message, sql[:256])

	except Exception as e:
		# Did we lose connection with the server?
		if e.args == (2006, 'MySQL server has gone away'):
			# Log a message
			print("execute: ERROR: " + \
								   "Lost connection with server...attempting" + \
								   "to reconnect")

		# Did we try to insert a duplicate entry in a unique index?
		elif e.args[0] == 1062:
				# Log an error in the log file
			print("execute: ERROR: " + \
				 "Could not rollback transaction: " + sql)
			return "Duplicate"

		# Different MySQL error...
		else:
			# Log an error in the log file
			print("execute: ERROR: " + \
					"Could not commit: " + sql)
	
		return False


def dbToList(dbResult, index=0):
	"""Convert a database query result into a list.

	Inputs:
	  dbResult = is the tupled data from a cursor.fetchall() statement
	  index = (optional) defaults to the first variable (0), but indicates
		  which columns to extract from the query result

	Returns:
	  resultList = a dictionary described above

	On an error, the function returns an empty list
	"""
	resultList = []

	if dbResult == []:
		return []

	# Check to make sure index is within range
	rowLength = len(dbResult)
	if rowLength == 0:
		return []
	colLength = len(dbResult[0])
	if index + 1 > colLength:
		return []

	for record in dbResult:
		resultList.append(record[index])

	return resultList

def dbToDict(dbResult, variableNames):
	"""Convert a database query result set (a tuple of tuples)
	to a dictionary where variable names are the dictionary
	keys and the dictionary values are lists of that variables
	data.

	Inputs:
	  dbResult = output from the SQL query (from cursor.fetchall())
	  variableNames = list of database table variable names in the
				same order as the results from the SQL query

	Returns:
	  dbDict = a dictionary where table variable names are keys to
			the dictionary and the values are the corresponding list
			of data for that variable
	"""
	dataDict = {}

	# Loop through each variable name and convert that column's
	#  data to list.  Then store that list in the dictionary
	for index, var in enumerate(variableNames):
		dataDict[var] = dbToList(dbResult, index)

	return dataDict

def getTableVars(cursor, database, table, getPrimaryKey=False):
	"""Query Information_Schema to get list of variables in this table
	By default, it excludes the Primary Key field, but can be captured
	by setting getPrimaryKey=True
	"""
	# Build the query
	sqlStr = "SELECT COLUMN_NAME " + \
			 " FROM information_schema.COLUMNS " + \
			 " WHERE TABLE_SCHEMA = '" + \
			 database + \
			 "' AND TABLE_NAME = '" + table + "'"

	# Make sure we don't get the primary key (if desired)
	if getPrimaryKey is False:
		sqlStr += " AND COLUMN_KEY != 'PRI'"

	# Execute the query
	cursor.execute(sqlStr)
	outputTuple = cursor.fetchall()

	return dbToList(outputTuple)


def _connect_():
	#Connect to MYSQL DATABASE
	import MySQLdb
	# Open database connection
	
	db = MySQLdb.connect("localhost","root","root","SarrahDataset")
	# prepare a cursor object using cursor() method
	cursor = db.cursor()
	return db,cursor




def loadFormSpringBullyingData(ipfilename):



	import MySQLdb 
	# Open database connection
	db = MySQLdb.connect("localhost","root","root","FormspringBullying")
	# prepare a cursor object using cursor() method
	cursor = db.cursor()

	url="file://"+ipfilename
	page = urllib2.urlopen(url,"lxml")
	f = open(ipfilename)
	xml = f.read()
	f.close()
	
	tree = etree.parse(StringIO(xml))
	# get root element
	root = tree.getroot()
	context = etree.iterparse(StringIO(xml))
	count=0
	resultlist=[]
	
	for item in root.findall('FORMSPRINGID'):
		count=count+1
		print count
		# empty news dictionary
		datadict = {}
		postdict={}
		labeldata=[]
		# iterate child elements of item
		for child in item:
			#print child.tag
			print child.tag
			if child.tag=="POST":
				cn=1
				postdict={}
				labeldata=[]
				for postdatanode in child:
					if(postdatanode.tag=="LABELDATA"):
						tempdict={}
						for labeldatanodecontentnode in postdatanode:
							if(labeldatanodecontentnode.text is None):
								tempdict[labeldatanodecontentnode.tag]=None
							else:
								tempdict[labeldatanodecontentnode.tag]=labeldatanodecontentnode.text.encode('utf8')
						labeldata.append(tempdict) 
					else:
						if(postdatanode.text is None):
							postdict[postdatanode.tag] = None
						else:
							postdict[postdatanode.tag] = postdatanode.text.encode('utf8')
						
				postdict["LABELDATA"]=  labeldata
					
				#insert it into database
				bio=str(datadict["BIO"]).replace("'","\\'")
				date=str(datadict["DATE"]).replace("'","\\'")
				location=str(datadict["LOCATION"]).replace("'","\\'")
				userid=str(datadict["USERID"]).replace("'","\\'")
				post=str(postdict["TEXT"])
				post=post.replace("'","\\'")
				asker=str(postdict["ASKER"]).replace("'","\\'")
				
					
					
				for itemdict in postdict["LABELDATA"]:
					
					answer=str(itemdict["ANSWER"]).replace("'","\\'")
					cyberbullyword=str(itemdict["CYBERBULLYWORD"]).replace("'","\\'")
					severity=str(itemdict["SEVERITY"]).replace("'","\\'")
					other=str(itemdict["OTHER"]).replace("'","\\'")
					worktime=str(itemdict["WORKTIME"]).replace("'","\\'")
					workerid=str(itemdict["WORKERID"]).replace("'","\\'")
					
					
					#Insert the record to database
					querySql="INSERT INTO `formspring_bullying_Labeled_data` (`bio`,`date`,`location`,`userid`,`text`,`asker`,`labelid`,`answer`,`cyberbullyword`,`severity`,`other`,`worktime`,`workerid`) VALUES ('"+ bio +"','"+ date +"','"+ location +"','"+ userid +"','"+ post +"','"+ asker +"','"+ str(cn) +"','"+ answer +"','"+ cyberbullyword +"','"+ severity+"','"+ other+"','"+ worktime+"','"+workerid+"')"
					cursor.execute(querySql)
					db.commit()
					cn=cn+1


			else:
				if child.text is None:
					datadict[child.tag] = None
				else:
					datadict[child.tag] = child.text.encode('utf8')

		resultlist.append(datadict)
	   
	
if __name__ == "__main__":
		
	ipfile="./formspring_Ip/XMLMergedFile.xml"
	reload(sys)
	sys.setdefaultencoding("utf-8")
	
	loadFormSpringBullyingData(ipfile)
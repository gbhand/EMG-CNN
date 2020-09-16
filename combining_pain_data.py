import csv
import os
import pandas as pd


current_directory= os.getcwd()

access_class= "Yes"

while access_class=="Yes":

	class_type= input("Enter folder name: ")
	final_path= current_directory + "/Pain_data/Data_raw/" + str(class_type) +"/"
	os.chdir(final_path)

	count=0

	for file in os.listdir(final_path):

		df= pd.read_csv(file,header=None)

		if count==1:
			df_final= pd.concat([df1,df])
			print(df_final)

		elif count>1:
			df_final= pd.concat([df_final,df])

		df1= df
		count+=1

	filename= str(class_type) + ".csv"
	df_final.to_csv(filename,sep=',', encoding='utf-8' )

	status= input("wanna continue: ")

	access_class= str(status)





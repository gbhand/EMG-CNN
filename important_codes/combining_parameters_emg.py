import pandas as pd
import numpy as np
import os


initial_path= "data_emg/EMG_data_1/"

for i in range(0,5):

	folder= input("Enter folder name: ")
	MEF_path= initial_path + str(folder) + "/MEF1.csv"
	MEF= np.genfromtxt(MEF_path,delimiter=',')
	MDF_path= initial_path + str(folder) + "/MDF1.csv"
	MDF= np.genfromtxt(MEF_path,delimiter=',')
	ARV_path= initial_path + str(folder) + "/ARV1.csv"
	ARV= np.genfromtxt(MEF_path,delimiter=',')
	RMS_path= initial_path + str(folder) + "/RMS1.csv"
	RMS= np.genfromtxt(MEF_path,delimiter=',')

	merged= np.column_stack((MEF, MDF, ARV, RMS))

	filename= str(folder) +"_all_parameters.csv"

	df= pd.DataFrame.from_records(merged)
	df.to_csv(filename, sep=',', encoding='utf-8')






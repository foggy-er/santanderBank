import pandas as pd 
import numpy as np
import random

##the function to shuffle the file with only one positive instance in a group.
def shuffleFile(filename,seednumber):
	##read in file 
	df_train=pd.read_csv(filename)

	####remove constant columns
	col_rm=[]
	for col in df_train.columns:
		if len(np.unique(df_train[col].values))==1:
			col_rm.append(col)



	df_train.drop(col_rm,axis=1,inplace=True)

	##Remove duplicated columns
	col_rm=[]
	columns=df_train.columns
	for i in range(len(columns)-1):
		temp =df_train[columns[i]].values
		for j in range(i+1,len(columns)):
			if np.array_equal(temp,df_train[columns[j]].values):
				col_rm.append(columns[j])

	df_train.drop(col_rm,axis=1,inplace=True)


	total_Train = df_train.drop(['ID'], axis=1).values
	Y_Train = df_train['TARGET'].values

	###get the variable whose target value is positive
	positive=1
	positive_train=total_Train[(Y_Train==positive)]
	##use nonzero because the positive is 1. 
	positive_index=np.nonzero(Y_Train)
	negative_train=np.delete(total_Train,positive_index,axis=0)



	###form output list
	output=[]
	group_length=len(negative_train)/len(positive_train)
	

	##divide the group equally 
	for i in range(len(positive_train)):
		group=[]
		random_index=random.sample(range(1,len(negative_train)),group_length)
		group.append(positive_train[i])
		group.extend(negative_train[random_index])
		negative_train=np.delete(negative_train,random_index,axis=0)
		output.append(group)

	##process the leftover instance
	random_index=random.sample(range(1,len(positive_train)),len(negative_train))

	for i in random_index:
		output[i].append(negative_train[0])
		negative_train=np.delete(negative_train,0,axis=0)


	return output








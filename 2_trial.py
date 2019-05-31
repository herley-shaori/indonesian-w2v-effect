import pandas

# function to remove non-ASCII
def remove_non_ascii(text):
	return ''.join(i for i in text if ord(i)<128)

lineList = [line.rstrip('\n') for line in open('600_news_data.csv',encoding='latin-1')]
fixedLineList=[]
for line in lineList:
	line=remove_non_ascii(line)
	fixedLineList.append(line)

totalKalimat=''
jumlahDokumen=0
dictList=[]
for x in range(1,len(fixedLineList)):
	if(';Hoax' in fixedLineList[x] or ';Valid' in fixedLineList[x]):
		totalKalimat+=fixedLineList[x]+' '
		jumlahDokumen+=1
		if(';Hoax' in fixedLineList[x]):
			dictList.append({'text':totalKalimat,'class':0})
		else:
			dictList.append({'text':totalKalimat,'class':1})
		totalKalimat=''
		#print(totalKalimat)
		#exit()
	else:
		totalKalimat+=fixedLineList[x]+' '
print('Jumlah Dokumen: ',jumlahDokumen)
print('LenDict: ', len(dictList))

df=pandas.DataFrame(dictList)
df.to_csv('news_data.csv',index=False)

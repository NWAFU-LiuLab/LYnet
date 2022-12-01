f1=open("Train.FASTA","r")
train_data=f1.readlines()
f1.close()
f2=open("Test.FASTA","r")
test_data=f2.readlines()
f2.close()

f1=open("positive.txt","w")
f2=open("negative.txt","w")
for i in range(len(train_data)):
    if i%2==0:
        if "positive" in train_data[i]:
            f1.write(train_data[i]+train_data[i+1])
        else:
            f2.write(train_data[i]+train_data[i+1])
            
for i in range(len(test_data)):
    if i%2==0:
        if "positive" in test_data[i]:
            f1.write(test_data[i]+test_data[i+1])
        else:
            f2.write(test_data[i]+test_data[i+1])            
f1.close()
f2.close()
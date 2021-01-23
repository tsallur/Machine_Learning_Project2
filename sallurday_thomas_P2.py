# Thomas Sallurday
# Professor Hodges
# CPSC 6430
# 10/5/2020
import numpy as np
import matplotlib.pyplot as plot

#hypoth function used for calculating predictions
def hypothFunc(data,weights,rows):
    hypoth = np.zeros([rows,1])
    for i in range(rows):
        step1 = -(np.dot(weights.T,data[i]))
        step2 = np.exp(step1)
        ans = 1/(1 + step2)
        hypoth[i] = ans
    return hypoth
#cost function used for calculating overall cost
def costFunc(hypoth,yVector,rows):
    cost = np.zeros([rows,1])
    O = np.ones([1,rows])
    for i in range(rows):
        cost[i] = -(yVector[i] * np.log(hypoth[i])) - ((1 - yVector[i]) * (np.log(1-hypoth[i])))
    totalCost = np.dot(O,cost)
    return (totalCost / rows)
# function to calculate what the new weights are
def newWeights(hypoth,yVector,data,weights,alpha):
    step1 = np.subtract(hypoth,yVector).T
    step2 = np.dot(step1,data).T
    step3 = step2 * (alpha / rows)
    return np.subtract(weights,step3)

# I found it easiest to go ahead and calculate and store the new features
# in a different text file. Coded under the assumption that the original file
# only has 2 features
str1 = input("Please enter the filename of the training set file: ")
openF = open(str1,'r')
fileName = "features.txt"
writeF = open(fileName,'w')
str1 = openF.readline()
str1 = str1.split('\t')
rows = int(str1[0])
cols = int(str1[1])
writeF.write(str(rows) + "\t" + "10" + "\n")

for i in range(rows): # nested for loop puts data in 2d array
    str1 = openF.readline()
    line = str1.split("\t")
    x1 = float(line[0])
    x2 = float(line[1])
    y = float(line[2])
    first = x1
    second = x2
    third = x2 * x2 * x2 * x2
    fourth = x1 * x1
    fifth = x1 * (x2) 
    sixth = x2 * (x1 * x1 * x1)
    seventh = (x2 * x2 * x2) * x1
    eighth = (x1 * x1 * x1 * x1) * (x2 * x2 * x2)
    ninth = (x1 * x1 * x1 * x1)
    tenth = (x2 * x2 * x2 * x2)
    writeF.write(str(first) + "\t" + str(second) + "\t" + str(third) + "\t" + str(fourth) + "\t" + str(fifth) + "\t" + str(sixth) + "\t" + str(seventh) + "\t" + str(eighth) +  "\t" + str(ninth) +"\t" + str(tenth) + "\t" + str(y) + "\n")
# close files after use
writeF.close()
openF.close()


trainData = open(fileName,'r')
str1 = trainData.readline()
str1 = str1.split('\t')
rows = int(str1[0])
cols = int(str1[1])
ogData = np.zeros([rows, cols]) #stores original data
yValVector = np.zeros([rows,1]) #stores y values
alpha = 0.95 # big alpha value seemed to work best
for i in range(rows): # nested for loop puts data in 2d array
    str1 = trainData.readline()
    line = str1.split("\t")
    for j in range(cols + 1):
        if(j == cols):
            yValVector[i] = float(line[j])
        else:
            ogData[i][j] = float(line[j])
trainData.close()
weights = np.zeros([cols,1])
hypoth = np.zeros([rows,1])
for i in range(cols):
    weights[i] = np.average(ogData[i].T) #initital weights are just averages
idealSize = 2000
# for loop plots iterations vs cost
for i in range(idealSize):
    hypoth = hypothFunc(ogData,weights,rows)
    cost = costFunc(hypoth,yValVector,rows)
    plot.scatter(i,cost,color = "orange",marker = "x",)
    weights = newWeights(hypoth,yValVector,ogData,weights,alpha)      
plot.xlabel("Number of Iterations")
plot.ylabel("J")
plot.title("Calculating Best Weights Plot")
plot.show()
        

str1 = input("Please enter the filename of the test set file: ")
openF = open(str1,'r')
fileName = "test.txt"
writeF = open(fileName,'w')
str1 = openF.readline()
str1 = str1.split('\t')
rows = int(str1[0])
cols = int(str1[1])
writeF.write(str(rows) + "\t" + "10" + "\n")

# same process as previous feature for loop
for i in range(rows): # nested for loop puts data in 2d array
    str1 = openF.readline()
    line = str1.split("\t")
    x1 = float(line[0])
    x2 = float(line[1])
    y = float(line[2])
    
    first = x1
    second = x2
    third = x2 * x2 * x2 * x2
    fourth = x1 * x1
    fifth = x1 * (x2) 
    sixth = x2 * (x1 * x1 * x1)
    seventh = (x2 * x2 * x2) * x1
    eighth = (x1 * x1 * x1 * x1) * (x2 * x2 * x2)
    ninth = (x1 * x1 * x1 * x1)
    tenth = (x2 * x2 * x2 * x2)
    writeF.write(str(first) + "\t" + str(second) + "\t" + str(third) + "\t" + str(fourth) + "\t" + str(fifth) + "\t" + str(sixth) + "\t" + str(seventh) + "\t" + str(eighth) +  "\t" + str(ninth) +"\t" + str(tenth) + "\t" + str(y) + "\n")
        
writeF.close()
openF.close()




inData = open(fileName,"r")
str1 = inData.readline()
str1 = str1.split('\t')
rows2 = int(str1[0])
cols2 = int(str1[1])
ogData2 = np.zeros([rows2, cols2])
yValVector2 = np.zeros([rows2,1])
for i in range(rows2):
    str1 = inData.readline()
    line = str1.split("\t")
    for j in range(cols2 + 1):
        if(j == cols2):
            yValVector2[i] = float(line[j])
        else:
            ogData2[i][j] = float(line[j])
inData.close()
myPredictions = np.zeros([rows2,1])
myPredictions = hypothFunc(ogData2,weights,rows2)
totCost = costFunc(myPredictions,yValVector2,rows2)
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(rows2):
    if (myPredictions[i] >= 0.5 and yValVector2[i] == 1):
        TP = TP + 1
    elif (myPredictions[i] < 0.5 and yValVector2[i] == 0):
        TN = TN + 1
    elif (myPredictions[i] >= 0.5 and yValVector[i] == 0):
        FP = FP + 1
    else:
        FN = FN + 1
print("Final J value: " + str(totCost))
print("Amount of true positives: " + str(TP))
print("Amount of true negatives " + str(TN))
print("Amount of false positives: " + str(FP))
print("Amount of false negatives: " + str(FN))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP/ (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (1 / ((1/precision) + (1/recall)))

print("Accuracy = " + str(accuracy))
print("Precision = " + str(precision))
print("Recall = " +str(recall))
print("F1 value: " + str(F1))

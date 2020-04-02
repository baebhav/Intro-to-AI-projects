#import neccessary libraries
import numpy as np
import math as m
from random import gauss
#create training and testing data from provided grids. 0 represents white cell, 1 represents black cell
trainingx=[]
trainingy=[]
testingx=[]
trainingx.append(np.array([[1,1,0,0,0],
                           [0,1,1,0,0],
                           [0,1,0,0,0],
                           [1,0,0,0,0],
                           [0,0,0,0,0]]))
trainingy.append(np.array([0]))
trainingx.append(np.array([[1,1,0,0,0],
                           [0,0,1,0,0],
                           [1,0,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,0,0]]))
trainingy.append(np.array([0]))
trainingx.append(np.array([[0,1,0,0,0],
                           [1,1,0,0,0],
                           [1,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,0,0,0]]))
trainingy.append(np.array([0]))
trainingx.append(np.array([[1,1,0,0,0],
                           [1,0,1,0,0],
                           [0,1,0,1,0],
                           [0,0,0,0,0],
                           [0,0,1,0,0]]))
trainingy.append(np.array([0]))
trainingx.append(np.array([[0,1,1,0,0],
                           [1,1,0,0,1],
                           [0,1,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,0,0]]))
trainingy.append(np.array([0]))
trainingx.append(np.array([[0,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,0,1,0],
                           [0,0,0,0,1],
                           [0,0,1,1,1]]))
trainingy.append(np.array([1]))
trainingx.append(np.array([[0,0,0,0,0],
                           [1,0,0,0,0],
                           [0,0,0,0,0],
                           [1,0,1,0,1],
                           [0,1,0,1,1]]))
trainingy.append(np.array([1]))
trainingx.append(np.array([[0,1,0,0,0],
                           [0,0,0,1,0],
                           [1,0,0,1,0],
                           [0,0,0,1,1],
                           [0,0,0,1,0]]))
trainingy.append(np.array([1]))
trainingx.append(np.array([[1,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,1,1],
                           [0,1,0,1,0]]))
trainingy.append(np.array([1]))
trainingx.append(np.array([[0,0,0,0,0],
                           [0,1,0,0,1],
                           [0,0,0,0,1],
                           [0,0,0,0,0],
                           [0,0,1,1,1]]))
trainingy.append(np.array([1]))
testingx.append(np.array([[1,0,0,0,0],
                           [1,0,1,0,1],
                           [0,0,0,0,0],
                           [0,0,1,1,1],
                           [0,0,0,1,0]]))
testingx.append(np.array([[1,1,1,0,0],
                           [1,1,1,0,0],
                           [0,0,1,1,0],
                           [0,0,0,0,0],
                           [1,0,1,0,0]]))
testingx.append(np.array([[0,0,0,0,1],
                           [0,0,0,0,1],
                           [0,0,0,1,1],
                           [0,0,0,0,1],
                           [0,1,0,1,1]]))
testingx.append(np.array([[0,1,1,0,0],
                           [1,1,0,0,0],
                           [0,1,1,0,0],
                           [0,1,0,0,1],
                           [0,0,0,0,0]]))
testingx.append(np.array([[0,1,1,0,0],
                           [1,1,0,0,0],
                           [0,1,1,0,0],
                           [0,1,0,0,1],
                           [0,0,0,0,0]]))
testingx.append(np.array([[1,0,0,0,1],
                           [0,0,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,0,0],
                           [1,0,0,0,1]]))
#transformation functions that complete different operations on the grids to create more training data
def rot90(grid):
    newgrid=np.zeros((5,5),dtype=np.float16)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[i][j]=grid[4-j][i]
    return newgrid
def rot180(grid):
    newgrid=np.zeros((5,5),dtype=np.float16)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[i][j]=grid[4-i][4-j]
    return newgrid

def rot270(grid):
    newgrid=np.zeros((5,5),dtype=np.float16)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[i][j]=grid[j][i]
    return newgrid

def flipx(grid):
    newgrid=np.zeros((5,5),dtype=np.float16)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[i][j]=grid[4-i][j]
    return newgrid
    
def flipy(grid):
    newgrid=np.zeros((5,5),dtype=np.float16)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            newgrid[i][j]=grid[i][4-j]
    return newgrid
#apply the transformation functions to the existing data and add that new training data to the current training data
def createnewtrainingdata(trainingdatax,trainingdatay):
    newtrainingdatax=[]
    newtrainingdatay=[]
    for i in range(len(trainingdatax)):
        newtrainingdatax.append(rot90(trainingdatax[i]))
        newtrainingdatax.append(rot180(trainingdatax[i]))
        newtrainingdatax.append(rot270(trainingdatax[i]))
        newtrainingdatax.append(flipx(trainingdatax[i]))
        newtrainingdatax.append(flipy(trainingdatax[i]))
        for j in range(5):
            newtrainingdatay.append(trainingdatay[i])
        
    trainingdatax=trainingdatax+newtrainingdatax
    trainingdatay=trainingdatay+newtrainingdatay
    return(trainingdatax,trainingdatay)
        
        
#train the neural network with training data      
def trainnn(trainingx,trainingy,gens):
    #initialize weights using Xavier initialization
    weights0=np.zeros((5,25),dtype=np.float16)
    for i in range(weights0.shape[0]):
        for j in range(weights0.shape[1]):
            weights0[i][j]=gauss(0,m.sqrt(1/weights0.shape[1]))
        
    bias0=1
    bias0weight=0
    weights1=np.zeros((5,5),dtype=np.float16)
    for i in range(weights1.shape[0]):
        for j in range(weights1.shape[1]):
            weights1[i][j]=gauss(0,m.sqrt(1/weights1.shape[1]))
    bias1=1
    bias1weight=0
    weights2=np.zeros((1,5),dtype=np.float16)
    for i in range(weights2.shape[0]):
        for j in range(weights2.shape[1]):
            weights2[i][j]=gauss(0,m.sqrt(1/weights2.shape[1]))
    bias2=1
    bias2weight=0
    #train the neural network on the cells in each grid
    for g in range(gens):
        for x in range(0,len(trainingx)):
            inputlayer=np.array([])
            for i in range(trainingx[x].shape[0]):
                for j in range(trainingx[x].shape[1]):
                    inputlayer=np.append(inputlayer,trainingx[x][i][j])
            classlabel=trainingy[x]
            outputlayer=np.zeros((1,1),dtype=np.float16)
            #pass forward and update activation values in nodes
            hiddenlayer0=np.zeros((1,5),dtype=np.float16)
            hiddenlayer1=np.zeros((1,5),dtype=np.float16)
            feedforward(inputlayer,hiddenlayer0,weights0,bias0,bias0weight)
            feedforward(hiddenlayer0,hiddenlayer1,weights1,bias1,bias1weight)
            feedforward(hiddenlayer1,outputlayer,weights2,bias2,bias2weight)
            #compute derivatives of loss with respect to the output node: a binary 0 or 1. 0 represents Class A
            #and 1 represents Class B
            dCdO=2*(outputlayer[0]-classlabel[0])
            dZ=lambda z:1/(1+m.exp(-z))*(1-(1/(1+m.exp(-z)))) #derivative of sigmoid function
            #derivative of ReLu activation function
            def dReLu(activation):
                if activation<0:
                    return 0
                else:
                    return 1
                
            learningrate=0.02
            #derative of output node with respect to ReLu activation function
            dOdZ=dReLu(lineartransform(hiddenlayer1,weights2[0])+bias2*bias2weight) 
            
            weights2[0] -= (dCdO*dOdZ*hiddenlayer1[0])*learningrate
            #bias has activation unit of 1 so dNode/dW=1
            bias2weight -= (dCdO*dOdZ)*learningrate
            #backpropogate from hiddenlayer1 to hiddenlayer0 and update weights1
            for j in range(weights1.shape[0]):
                for k in range(weights1.shape[1]):
                    weights1[j][k] -= ((dCdO*dOdZ*weights2[0][j])*dReLu(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight)*hiddenlayer0[0][k])*learningrate
            bias1weight -= ((dCdO*dOdZ*weights2[0][j])*dReLu(lineartransform(hiddenlayer0,weights1[j])+bias1*bias1weight))*learningrate
                
                    
            #backpropogate from hiddenlayer0 to input layer and update weights0
            #The gradient for each node in hiddenlayer1 needs to be summed because 
            #there are multiple routes to the output node through each one of the nodes
            for j in range(weights0.shape[0]):
                for k in range(weights0.shape[1]):
                    gradient=0
                    for z in range(hiddenlayer1.shape[1]):
                        gradient += ((dCdO*dOdZ*weights2[0][z]))*dReLu(lineartransform(hiddenlayer0,weights1[z])+bias1*bias1weight)*weights1[z][j]
                        
                    biasgradient=gradient*dReLu(lineartransform(inputlayer,weights0[j])+bias0*bias0weight)
                    gradient=gradient*dReLu(lineartransform(inputlayer,weights0[j])+bias0*bias0weight)*inputlayer[k] 
                    
                    weights0[j][k] -= (gradient)*learningrate 
            bias0weight -= (biasgradient)*learningrate
    
    return [weights0,bias0weight,weights1,bias1weight,weights2,bias2weight] 
#function that updates node values for next layer
def feedforward(layer0,layer1,weights,bias,biasweight):
    for i in range(layer1.shape[1]):
        layer1[0][i]=ReLu(lineartransform(layer0,weights[i])+bias*biasweight)
def sigmoid(activation):
    return 1/(1+m.exp(-activation))
#rectified linear unit activation function
def ReLu(activation):
    return max(0,activation)
#matrix multiplication to update node values
def lineartransform(inputlayer,weights):
    return np.matmul(inputlayer,np.transpose(weights))
def nnpredict(testingdatax,testingdatay,model):
    weights0=model[0]
    bias0weight=model[1]
    weights1=model[2]
    bias1weight=model[3]
    weights2=model[4]
    bias2weight=model[5]
    predictions=[]
    for x in range(len(testingdatax)):
        classlabel=testingdatay[x]
        inputlayer=np.array([])
        outputlayer=np.zeros((1,1),dtype=np.float16)
        for i in range(5):
            for j in range(5):
                inputlayer=np.append(inputlayer,testingdatax[x][i][j])
        hiddenlayer0=np.zeros((1,5),dtype=np.float16)
        hiddenlayer1=np.zeros((1,5),dtype=np.float16)
        feedforward(inputlayer,hiddenlayer0,weights0,1,bias0weight)
        feedforward(hiddenlayer0,hiddenlayer1,weights1,1,bias1weight)
        feedforward(hiddenlayer1,outputlayer,weights2,1,bias2weight)
        #predictions.append(outputlayer[0])
        if outputlayer[0]<0.5:
            predictions.append(np.array([0]))
        else:
            predictions.append(np.array([1]))
    #return predictions
    wrong=0
    for i in range(len(predictions)):
        if abs(predictions[i][0]-testingdatay[i][0])==1:
            wrong += 1
    accuracy=(len(predictions)-wrong)/len(predictions)
    return accuracy
#intialize node objects that will make up the decision tree
#each node has feature and output data associated with it 
class Node(object):
    def __init__(self,val,varindex,datax,datay,bannedlst):
        self.val=val
        self.left=None
        self.right=None
        self.varindex=varindex #keep track of the variable associated with the node
        self.datax=datax
        self.datay=datay
        self.returnlabel=None
        self.bannedlst=bannedlst#updates to a value if the entire path down the tree from the node
                              #has an output of 0 or 1. 
    #check if the all the data associated with the node contains only one output label
    def getlabel(self):
        all1=True
        all0=True
        for i in self.datay:
            if i[0] != 1:
                all1=False
            if i[0] != 0:
                all0=False
        if all1:
            return 1
        if all0:
            return 0
        return 'None'
#create decision tree and insert statement that helps build out the tree
class DecisionTree(object):
    def __init__(self,root):
        self.root=root
    def insert(self,node,val,varindex,datax,datay,bannedlst):
        if val==1:
            node.right=Node(val,varindex,datax,datay,bannedlst)
        else:
            node.left=Node(val,varindex,datax,datay,bannedlst)
    
    def returnroot(self):
        return self.root
        
            
#function to build the decision tree using the given training data
def builddecisiontree(trainingx,trainingy):
    tree=DecisionTree(Node(0,(-1,-1),trainingx,trainingy,[]))
    lst=[tree.root]
    i=0
    while len(lst)>0:
        print(i)
        node=lst.pop()
        #generates the data for each branch leaving the node. Right branch represents an xi value
        #of 1 and left branch represents a value of 0.
        nextlevel=movedown(node.datax,node.datay,node.bannedlst)
        varindex=nextlevel[4]
        #if the outputs are ambiguous through the rest of the tree, keep building
        if node.getlabel() == 'None':
            tree.insert(node,0,varindex,nextlevel[0],nextlevel[1],node.bannedlst+[varindex])
            tree.insert(node,1,varindex,nextlevel[2],nextlevel[3],node.bannedlst+[varindex])
            lst.append(node.right)
            lst.append(node.left)
        #else, stop building and set node's output value
        else:
            if node.getlabel()==1:
                node.returnlabel=1
            if node.getlabel()==0:
                node.returnlabel=0
        i += 1
    return tree
#function that predicts the output labels for all the training examples 
def predictdd(tree,testingx,testingy):
    predictions=[]
    for i in range(len(testingx)):
        root=tree.returnroot()
        lst=[root]
        while len(lst)>0:
            node=lst.pop()
            #determine the variable we need to check for a the node
            xvar=node.varindex
            #if the node is providing an returnlabel, then we know the output
            #must be that number and we can stop searching the tree
            if node.returnlabel==1:
                predictions.append(np.array([1]))
                break
            if node.returnlabel==0:
                predictions.append(np.array([0]))
                break
            #if the value of the variable in the training example is 1, move down the right branch
            #and if its 0 then move down the left branch
            if testingx[i][xvar[0]][xvar[1]]==1:
                lst.append(node.right)
            if testingx[i][xvar[0]][xvar[1]]==0:
                lst.append(node.left)
    wrong=0
    #for i in range(len(predictions)):
        #if abs(predictions[i][0]-testingy[i][0])==1:
            #wrong += 1
    #return (len(testingx)-wrong)/len(testingx)
    printpredictions
            
#function that moves down the tree and creates data for new branches. This function identifies
#the variable with the highest information gain and partitions the data according to that variable. The
#output from this function is sent to the builddecisiontree function. 
def movedown(trainingx,trainingy,bannedlst):
    splitvar=informationgain(trainingx,trainingy,bannedlst)
    zerotrainingx=[]
    zerotrainingy=[]
    onetrainingx=[]
    onetrainingy=[]
    for i in range(len(trainingx)):
        if trainingx[i][splitvar[0]][splitvar[1]]==0:
            zerotrainingx.append(trainingx[i])
            zerotrainingy.append(trainingy[i])
        else:
            onetrainingx.append(trainingx[i])
            onetrainingy.append(trainingy[i])
    return (zerotrainingx,zerotrainingy,onetrainingx,onetrainingy,splitvar)
        
#Uses the ID3 algorithm from the Decision Tree notes to output a list containing 
#the information gains associated with each variable given the input data.
def log(n):
        if n>0:
            return m.log2(n)
        else:
            return 0
def informationgain(trainingx,trainingy,bannedlst):
    informationgainx=[]
    variables=[]
    for i in range(trainingx[0].shape[0]):
        for j in range(trainingx[0].shape[1]):
            if (i,j) not in bannedlst:
                variables.append((i,j))
                x0count=0
                x1count=0
                y0x0count=0
                y0x1count=0
                y1x0count=0
                y1x1count=0
                for k in range(len(trainingx)):
                    if trainingx[k][i][j]==0:
                        x0count += 1
                    else:
                        x1count += 1
                    if trainingy[k][0]==0 and trainingx[k][i][j]==0:
                        y0x0count += 1
                    if trainingy[k][0]==0 and trainingx[k][i][j]==1:
                        y0x1count += 1
                    if trainingy[k][0]==1 and trainingx[k][i][j]==0:
                        y1x0count += 1
                    if trainingy[k][0]==1 and trainingx[k][i][j]==1:
                        y1x1count += 1
            
                if x0count==0:
                    px0=0
                    py0x0=0
                    py1x0=0
                else:
                    px0=x0count/len(trainingx)
                    py0x0=y0x0count/x0count
                    py1x0=y1x0count/x0count
                if x1count==0:
                    px1=0
                    py0x1=0
                    py1x1=0
                else:
                    px1=x1count/len(trainingx)
                    py0x1=y0x1count/x1count
                    py1x1=y1x1count/x1count
                
                coninfcontent=px0*-((py0x0*log(py0x0))+(py1x0*log(py1x0)))+px1*-((py0x1*log(py0x1))+(py1x1*log(py1x1)))
            
                informationgainx.append(coninfcontent)
    return variables[informationgainx.index(min(informationgainx))]
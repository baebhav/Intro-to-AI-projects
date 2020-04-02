import random
import matplotlib.pyplot as plt
#initialize cell object used in board with terrains
class Cell(object):
    def __init__(self,loc,terrain,fn):
        self.loc=loc
        self.target=False
        self.terrain=terrain
        self.fn=fn
#initialize cell object used in board with corresponding target belief state 
class probCell(object):
    def __init__(self,loc,p):
        self.p=p
        self.loc=loc
#creates board of different terrains using random number generator to simulate probability distribution of
#terrain types
def createphysicalboard(n):
    board=[]
    for i in range(n):
        row=[]
        for j in range(n):
            num=random.randint(0,99)
            if num>=0 and num<=19:
                row.append(Cell((i,j),'flat',0.1))
            if num>=20 and num<=49:
                row.append(Cell((i,j),'hilly',0.3))
            if num>=50 and num<=79:
                row.append(Cell((i,j),'forested',0.7))
            if num>=80 and num<=99:
                row.append(Cell((i,j),'maze',0.9))
        board.append(row)
    #assign target to random cell
    (num1,num2)=(random.randint(0,n-1),random.randint(0,n-1))
    board[num1][num2].target=True
    return board
#creates board with corresponding belief that given cell is the target. Belief is equally initialized at 
#1/n^2 for every cell 
def createprobboard(n):
    board=[]
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(probCell((i,j),1/n**2))
        board.append(row)
    return board
#algorithm that simulates searching board for target
def searchanddestroy(mainboard,pboard):
    gameover=False
    searches=0
    while not gameover:
        #obtain the location of the cell with the highest belief state
        cellloc=findmaxcell(pboard,mainboard)
        searches += 1
        #probability that target is in current cell and searching it will result in the target not being found
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        #probability that searching the current cell will result in the target not being found
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        #update probability of current cell using Baye's Theorem: P(target in current cell | target wasn't found in current cell)=
        #(P(probtarget)/(probnottarget))
        pboard[cellloc[0]][cellloc[1]].p=probtarget/probnottarget
        #if cell contains the target, simulate the false negative rate using random number generator 
        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return searches
        pupdate(pboard,mainboard,probnottarget,cellloc)
#same exact algorithm as normal search algorithm except a call is made to the function findmaxcellfn which incorporates
#false negative rate in updating belief states
def searchanddestroyfn(mainboard,pboard):
    gameover=False
    searches=0
    while not gameover:
        cellloc=findmaxcellfn(pboard,mainboard)
        searches += 1
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        pboard[cellloc[0]][cellloc[1]].p=(probtarget/(1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))))
        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return searches
        pupdate(pboard,mainboard,probnottarget,cellloc)
        
        
#update belief state of every cell      
def pupdate(pboard,mainboard,probnottarget,queryloc):
    for i in pboard:
        for j in i:
            if j.loc != queryloc:
                #update probability of cell using Baye's Theorem: P(target is in cell | target wasn't found in cell just searched)=
                #P(target in current cell AND target wasn't found in searched cell)=P(target in current cell)/probnottarget
                pboard[j.loc[0]][j.loc[1]].p=(pboard[j.loc[0]][j.loc[1]].p/probnottarget)

#first stage of updating belief state after target has moved. The probability of a cell containing the target
#is equally distributed to all of its neighbors. We use a temporary probability map to redistribute these probabilities
#and then update 
def pupdatemoving1(mainboard,pboard):
    tempboard=[]
    for i in range(len(pboard)):
        row=[]
        for j in range(len(pboard)):
            row.append(0)
        tempboard.append(row)
    for i in range(len(tempboard)):
        for j in range(len(tempboard)):
            neighborloclst=[]
            if i+1<=len(tempboard)-1:
                neighborloclst.append((i+1,j))
            if i-1>=0:
                neighborloclst.append((i-1,j))
            if j+1<=len(tempboard)-1:
                neighborloclst.append((i,j+1))
            if j-1>=0:
                neighborloclst.append((i,j-1))
            for k in neighborloclst:
                tempboard[k[0]][k[1]] += (pboard[i][j].p)/len(neighborloclst)
    for i in range(len(pboard)):
        for j in range(len(pboard)):
            pboard[i][j].p=tempboard[i][j]
            
            
#second stage of updating cell probabilities after target has moved. Any cell that has the same 
#terrain as the clue terrain will have its belief state set to 0. The rest of the probabilities are updated
#using Baye's Theorem: P(Target in Celli | Target not in clue terrain)=P(Target in Celli and Target not in clue terrain)/P(Target not in clue terrain)
#=P(Target in Celli)/P(Target not in clue terrain)=P(Target in Celli)/(1-P(Target in clue terrain))
def pupdatemoving2(mainboard,pboard,terrainclue):
    cumprob=0
    for i in pboard:
        for j in i:
            if mainboard[j.loc[0]][j.loc[1]].terrain==terrainclue:
                cumprob += pboard[j.loc[0]][j.loc[1]].p
                pboard[j.loc[0]][j.loc[1]].p=0
    
    for k in pboard:
        for l in k:
            if mainboard[l.loc[0]][l.loc[1]].terrain != terrainclue:
                pboard[l.loc[0]][l.loc[1]].p=pboard[l.loc[0]][l.loc[1]].p/(1-cumprob)
                
                
#find cell with highest belief state 
def findmaxcell(pboard,mainboard):
    maxp=pboard[0][0].p
    maxloc=pboard[0][0].loc
    for i in pboard:
        for j in i:
            if j.p>maxp:
                maxp=j.p
                maxloc=j.loc
    return maxloc
#find cell with highest probability of finding target: P(finding target)=P(cell has target)*(1-false negative rate)           
def findmaxcellfn(pboard,mainboard):
    maxp=pboard[0][0].p*(1-mainboard[0][0].fn)
    maxloc=pboard[0][0].loc
    for i in pboard:
        for j in i:
            if j.p*(1-mainboard[j.loc[0]][j.loc[1]].fn)>maxp:
                maxp=j.p*(1-mainboard[j.loc[0]][j.loc[1]].fn)
                maxloc=j.loc
    return maxloc
#Uses the same probability of finding target in given cell as from the maxcellfn function. The distance from
#the searched cell to the current cell multiplied by a small weight is subtracted from that probability. If the
#weight is too large then the algorithm will search cells with low probabilities just because they are closer. If
#the weight is too small then the algorithm will search cells that are far away and have negligibly higher
#probabilities than close cells.
def findmaxcellheuristic(pboard,mainboard,cellloc,w):
    maxh=pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn)
    maxloc=pboard[cellloc[0]][cellloc[1]].loc
    for i in pboard:
        for j in i:
            if j.p*(1-mainboard[j.loc[0]][j.loc[1]].fn)-w*((abs(j.loc[0]-cellloc[0])+abs(j.loc[1]-cellloc[1]))/(len(mainboard)*2))>maxh:
                maxh=j.p*(1-mainboard[j.loc[0]][j.loc[1]].fn)-w*((abs(j.loc[0]-cellloc[0])+abs(j.loc[1]-cellloc[1]))/(len(mainboard)*2))
                maxloc=j.loc
    return maxloc



def resetp(pboard):
    for i in pboard:
        for j in i:
            pboard[j.loc[0]][j.loc[1]].p=1/(len(pboard)**2)

#modification on original algorithm. Tracks actions instead of searches. Searching a cell counts as an action and
#the total distance traveled(Manhatten distance) is added to the number of actions.
def searchanddestroyfnwithcost(mainboard,pboard):
    gameover=False
    actions=0
    previouscellloc=(0,0)
    while not gameover:
        cellloc=findmaxcellfn(pboard,mainboard)
        actions += abs(cellloc[0]-previouscellloc[0])+abs(cellloc[1]-previouscellloc[1])+1
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        pboard[cellloc[0]][cellloc[1]].p=(probtarget/(1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))))
        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return actions
        pupdate(pboard,mainboard,probnottarget,cellloc)
        previouscellloc=cellloc
        
def searchanddestroywithcost(mainboard,pboard):
    gameover=False
    actions=0
    previouscellloc=(0,0)
    while not gameover:
        cellloc=findmaxcell(pboard,mainboard)
        actions += abs(cellloc[0]-previouscellloc[0])+abs(cellloc[1]-previouscellloc[1])+1
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        pboard[cellloc[0]][cellloc[1]].p=(probtarget/(1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))))
        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return actions
        pupdate(pboard,mainboard,probnottarget,cellloc)
#incorporates a weight on distance between given cell and the cell that was just searched. Algorithm can get stuck
#in local maximum where it doesn't want to leave current cell so random restart is implemented. 
def searchanddestroyheuristicwithcost(mainboard,pboard,w):
    gameover=False
    actions=0
    previouscellloc=(0,0)
    counter=0
    while not gameover:
        cellloc=findmaxcellheuristic(pboard,mainboard,previouscellloc,w)
        if cellloc==previouscellloc:
            counter += 1
        else:
            counter = 0
        #if algorithm is stuck, randomly pick a new cell on the board
        if counter==5:
            previouscellloc=cellloc
            cellloc=(random.randint(0,len(mainboard)-1),random.randint(0,len(mainboard)-1))
            counter=0
            
        actions += abs(cellloc[0]-previouscellloc[0])+abs(cellloc[1]-previouscellloc[1])+1
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        pboard[cellloc[0]][cellloc[1]].p=(probtarget/(1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))))
        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return actions
        pupdate(pboard,mainboard,probnottarget,cellloc)
        previouscellloc=cellloc

def searchanddestroyheuristicwithcostmoving(mainboard,pboard,w):
    gameover=False
    actions=0
    previouscellloc=(0,0)
    counter=0
    while not gameover:
        cellloc=findmaxcellheuristic(pboard,mainboard,previouscellloc,w)
        if cellloc==previouscellloc:
            counter += 1
        else:
            counter = 0
        #if algorithm is stuck, randomly pick a new cell on the board
        if counter==5:
            previouscellloc=cellloc
            cellloc=(random.randint(0,len(mainboard)-1),random.randint(0,len(mainboard)-1))
            counter=0
            
        actions += abs(cellloc[0]-previouscellloc[0])+abs(cellloc[1]-previouscellloc[1])+1
        probtarget=pboard[cellloc[0]][cellloc[1]].p*mainboard[cellloc[0]][cellloc[1]].fn
        probnottarget=1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))
        pboard[cellloc[0]][cellloc[1]].p=(probtarget/(1-(pboard[cellloc[0]][cellloc[1]].p*(1-mainboard[cellloc[0]][cellloc[1]].fn))))

        if mainboard[cellloc[0]][cellloc[1]].target==True:
            inttargetfound=int((1-mainboard[cellloc[0]][cellloc[1]].fn)*100)
            randnum=random.randint(0,99)
            if randnum<=inttargetfound-1:
                return actions
        
        pupdate(pboard,mainboard,probnottarget,cellloc)
        #once the target is not found, it moves to a neighboring cell with equal probability. After the move,
        #there will be three terrains where the target is not located. One of these terrains is randomly chosen
        #to give the agent a clue which the terrain the target is not located in.
        neighborloclst=[]
        if cellloc[0]+1<=len(mainboard)-1:
            neighborloclst.append((cellloc[0]+1,cellloc[1]))
        if cellloc[0]-1>=0:
            neighborloclst.append((cellloc[0]-1,cellloc[1]))
        if cellloc[1]+1<=len(mainboard)-1:
            neighborloclst.append((cellloc[0],cellloc[1]+1))
        if cellloc[1]-1>=0:
            neighborloclst.append((cellloc[0],cellloc[1]-1))
                    
        newtargetloc=neighborloclst[random.randint(0,len(neighborloclst)-1)]
        mainboard[cellloc[0]][cellloc[1]].target=False
        mainboard[newtargetloc[0]][newtargetloc[1]].target=True
        terraintype=mainboard[newtargetloc[0]][newtargetloc[1]].terrain
        notterrains=[i for i in ['hilly','flat','forested','maze'] if i != terraintype]
        terrainclue=notterrains[random.randint(0,2)]
        #stage 1 probability update
        pupdatemoving1(mainboard,pboard)
        #stage 2 probability update
        pupdatemoving2(mainboard,pboard,terrainclue)
                
                
        previouscellloc=cellloc
#plots average number of searches taken by Rule 1 vs. Rule 2 
def plotrule1vsrule2():
    boarditer=[]
    searches=[]
    searchesfn=[]
    for i in range(10):
        boarditer.append(i+1)
        board=createphysicalboard(20)
        pboard=createprobboard(20)
        search=[]
        searchfn=[]
        for j in range(50):
            search.append(searchanddestroy(board,pboard))
            resetp(pboard)
            searchfn.append(searchanddestroyfn(board,pboard))
            for k in board:
                for s in k:
                    board[s.loc[0]][s.loc[1]].target=False
            board[random.randint(0,19)][random.randint(0,19)].target=True
            print(i,j)
        searches.append(sum(search)/len(search))
        searchesfn.append(sum(searchfn)/len(searchfn))
    plt.plot(boarditer,searches,label='Rule1')
    plt.plot(boarditer,searchesfn,label='Rule2')
    plt.xlabel('Board Iteration')
    plt.ylabel('Average Number of Searches')
    plt.legend()
    plt.show()
#Plots average number of actions taken using Rule 2 vs. Distance Heuristic
def plotheuristicvsfn():
    boarditer=[]
    searchesfn=[]
    searchesheuristic=[]
    for i in range(10):
        boarditer.append(i+1)
        board=createphysicalboard(20)
        pboard=createprobboard(20)
        searchfn=[]
        searchheuristic=[]
        for j in range(50):
            searchfn.append(searchanddestroyfnwithcost(board,pboard))
            resetp(pboard)
            searchheuristic.append(searchanddestroyheuristicwithcost(board,pboard,.005))
            for k in board:
                for s in k:
                    board[s.loc[0]][s.loc[1]].target=False
            board[random.randint(0,19)][random.randint(0,19)].target=True
            print(i,j)
        searchesfn.append(sum(searchfn)/len(searchfn))
        searchesheuristic.append(sum(searchheuristic)/len(searchheuristic))
        
    plt.plot(boarditer,searchesfn,label='Rule2')
    plt.plot(boarditer,searchesheuristic,label='Heuristic')
    plt.xlabel('Board Iteration')
    plt.ylabel('Average Number of Actions')
    plt.legend()
    plt.show()
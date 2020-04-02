import matplotlib.pyplot as plt
import random
from itertools import combinations 
import itertools
#standard priority queue imeplementation that compares probability of cells being mines
class Pqueue:
    def __init__(self):
        self.heap=[0]
        self.size=0
    
    def buildheap(self,lst):
        index=len(lst)//2
        self.size=len(lst)
        self.heap=[0]+lst[:]
        while index>0:
            self.movedown(index)
            index=index-1
    
    def insert(self,item):
        self.heap.append(item)
        self.size += 1
        self.moveup(self.size)
    
    def popmin(self):
        minval=self.heap[1]
        self.heap[1]=self.heap[self.size]
        self.size -= 1
        self.heap.pop()
        self.movedown(1)
        return minval
    
    def moveup(self,index):
        while index//2>0:
            if self.heap[index].p<self.heap[index//2].p:
                tmp=self.heap[index//2]
                self.heap[index//2]=self.heap[index]
                self.heap[index]=tmp
            index=index//2


    def movedown(self,i):
        while (2*i)<=self.size:
            minindex=self.minchild(i)
            if self.heap[i].p>self.heap[minindex].p:
                parent=self.heap[i]
                self.heap[i]=self.heap[minindex]
                self.heap[minindex]=parent
            i=minindex

    def minchild(self,index):
        if 2*index+1>self.size:
            return 2*index
        else:
            if self.heap[2*index].p<self.heap[2*index+1].p:
                return 2*index
            else:
                return 2*index+1
#Initialization of cell objects that make up the minesweeper board. Attributes include relationships to surrounding 
#cells and methods that return information about neighbor cells
class Cell(object):
    def __init__(self,mine):
        if mine==1:
            self.mine=True
        else:
            self.mine=False
        self.hidden=True
        self.identifiedsafe=False
        self.identifiedmine=False
        self.numneighbors=0
        self.hiddenneighbors=0
        self.numminesidentified=0
        self.numsafeidentified=0
        self.numminesaround=None
        self.location=(None,None)
        self.neighbors=[]
        self.locneighbors=[]
        self.visited=False
        self.p=0
        self.loc=(None,None)
        self.pupdated=False
        self.minecount=0
        self.loc=(None,None)
    def returnsafeidentified(self,board):
        counter=0
        for i in self.neighbors:
            if board[i.loc[0]][i.loc[1]].identifiedsafe==True:
                counter += 1
        return counter
    def returnminesidentified(self,board):
        counter=0
        for i in self.neighbors:
            if board[i.loc[0]][i.loc[1]].identifiedmine==True:
                counter += 1
        return counter
                
    def returnmineneighbors(self):
        counter=0
        for i in self.neighbors:
            if i.mine==True:
                counter += 1
        return counter
    def returnhiddenneighbors(self,board):
        counter=0
        for i in self.neighbors:
            if board[i.loc[0]][i.loc[1]].hidden==True:
                counter += 1
        return counter

#function that outputs the number of neighbor cells that a cell has based on its (i,j) location within the board and 
#the size of the board d
def numneighbors(i,j,d):
    if ((i-1)<0 and (j-1)<0) or ((i-1)<0 and (j+1)>=d) or ((i+1)>=d and (j-1)<0) or ((i+1)>=d and (j+1)>=d):
        return 3
    else:
        if (i-1)<0 or (i+1)>=d or (j-1)<0 or (j+1)>=d:
            return 5
        else:
            return 8
#Creates a list of lists containing Cell objects. Mines are randomly scattered around the board. As the cells are
#traversed, their relationship to neighbors is updated by creating pointers to neighbor cells in the neighbors list
def createboard(d,n):
    nummines=n
    board=[]
    minelst=[]
    for i in range(d):
        row=[]
        for j in range(d):
            row.append(Cell(0))
            minelst.append((i,j))
        board.append(row)
    random.shuffle(minelst)
    while nummines>0:
        minecell=minelst.pop()
        board[minecell[0]][minecell[1]].mine=True
        nummines -= 1
    
    for i in range(len(board)):
        for j in range(len(board)):
            board[i][j].numneighbors=numneighbors(i,j,len(board))
            board[i][j].loc=(i,j)
            if (i-1)>=0:
                board[i][j].top=board[i-1][j]
                board[i][j].neighbors.append(board[i-1][j])
            if (i+1)<=len(board)-1:
                board[i][j].bottom=board[i+1][j]
                board[i][j].neighbors.append(board[i+1][j])
            if (j+1)<=len(board)-1:
                board[i][j].right=board[i][j+1]
                board[i][j].neighbors.append(board[i][j+1])
            if (j-1)>=0:
                board[i][j].left=board[i][j-1]
                board[i][j].neighbors.append(board[i][j-1])
            if (i-1)>=0 and (j-1)>=0:
                board[i][j].topleft=board[i-1][j-1]
                board[i][j].neighbors.append(board[i-1][j-1])
            if (i-1)>=0 and (j+1)<=len(board)-1:
                board[i][j].topright=board[i-1][j+1]
                board[i][j].neighbors.append(board[i-1][j+1])
            if (i+1)<=len(board)-1 and (j-1)>=0:
                board[i][j].bottomleft=board[i+1][j-1]
                board[i][j].neighbors.append(board[i+1][j-1])
            if (i+1)<=len(board)-1 and (j+1)<=len(board)-1:
                board[i][j].bottomright=board[i+1][j+1]
                board[i][j].neighbors.append(board[i+1][j+1])

    return board
                
#Function for the baseline minesweeper agent. If there is a cell that has been identified as safe, the agent 
#will visit that cell. Otherwise, it will sequentially choose a cell that hasn't been visited. The strategies
#from the assignment instructions are implemented to uncover safe cells and mine cells without needing to visit
#every cell. The physical board is printed out after every move. 
def minesweeper(board):
    gameboard=[]
    boardlen=len(board)
    maxdigits=len(str(len(board)**2-1))
    for i in range(len(board)):
        row=[]
        for j in range(len(board)):
            row.append('['+'0'*(maxdigits-len(str(i*len(board)+j)))+str(i*len(board)+j)+']')
        gameboard.append(row)
    gameover=False
    cellsdetected=0
    minesidentified=0
    minesblown=0
    while not gameover:
        safecell=False
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].visited==False and board[i][j].identifiedsafe==True:
                    location=board[i][j].loc
                    safecell=True

        if safecell==False:
            for i in range(boardlen):
                for j in range(boardlen):
                    if board[i][j].visited==False and board[i][j].identifiedmine==False:
                        location=(i,j)  
        
        board[location[0]][location[1]].hidden=False
        board[location[0]][location[1]].visited=True
        cellsdetected += 1
        if board[location[0]][location[1]].mine==True:
            board[location[0]][location[1]].identifiedmine=True
            gameboard[location[0]][location[1]]="[  ]"
            minesblown += 1
            
        else:
            board[location[0]][location[1]].identifiedsafe=True
            board[location[0]][location[1]].numminesaround=board[location[0]][location[1]].returnmineneighbors()
            gameboard[location[0]][location[1]]="["+"S"+str(board[location[0]][location[1]].numminesaround)+"]"
            
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].identifiedsafe and board[i][j].visited==True:
                    board[i][j].numsafeidentified=board[i][j].returnsafeidentified(board)
                    if board[i][j].numminesaround-board[i][j].returnminesidentified(board)==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedmine=True
                                gameboard[k.loc[0]][k.loc[1]]='[  ]'
                                minesidentified += 1
                                cellsdetected += 1
                                board[k.loc[0]][k.loc[1]].hidden=False
    
                    if (board[i][j].numneighbors-board[i][j].numminesaround)-board[i][j].numsafeidentified==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedsafe=True
                                board[k.loc[0]][k.loc[1]].hidden=False
    
        if cellsdetected==len(board)**2:
            gameover=True
        
    return minesidentified/(minesblown+minesidentified)
#Instead of sequentially visiting cells that haven't been identified as safe, this modified agent calculates 
#probability of a cell being a mine for every cell on the board. The cell with the lowest probabiltiy is visited and the
#board is scanned after every move to update the probabilities. This function prints the current state of the board
#after every move
def Astarminesweeper(board):
    gameboard=[]
    maxdigits=len(str(len(board)**2-1))
    boardlen=len(board)
    for i in range(boardlen):
        row=[]
        for j in range(boardlen):
            row.append('['+'0'*(maxdigits-len(str(i*boardlen+j)))+str(i*boardlen+j)+']')
        gameboard.append(row)
    gameover=False
    cellsdetected=0
    minesidentified=0
    minesblown=0
    safefound=0
    for i in gameboard:
        print(*i)
    print('')
        
    while not gameover: 
        safecell=False
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].visited==False and board[i][j].identifiedsafe==True:
                    location=board[i][j].loc
                    safecell=True
                    

        if safecell==False:
            lst=[]
            for i in range(boardlen):
                for j in range(boardlen):
                    if board[i][j].hidden==True:
                        lst.append(board[i][j]) 
            pq=Pqueue()
            pq.buildheap(lst)
            location=pq.popmin().loc
        if board[location[0]][location[1]].hidden==True:
            cellsdetected += 1
        board[location[0]][location[1]].hidden=False
        board[location[0]][location[1]].visited=True
        if board[location[0]][location[1]].mine==True:
            board[location[0]][location[1]].identifiedmine=True
            gameboard[location[0]][location[1]]="[  ]"
            print('mine blown')
            minesblown += 1
            
            
        else:
            board[location[0]][location[1]].identifiedsafe=True

            board[location[0]][location[1]].numminesaround=board[location[0]][location[1]].returnmineneighbors()
            gameboard[location[0]][location[1]]="["+"S"+str(board[location[0]][location[1]].numminesaround)+"]"
            if board[location[0]][location[1]].hidden==True:
                safefound += 1
    
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].identifiedsafe==True and board[i][j].visited==True:
                    board[i][j].numsafeidentified=board[i][j].returnsafeidentified(board)
                    if board[i][j].numminesaround-board[i][j].returnminesidentified(board)==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedmine=True
                                gameboard[k.loc[0]][k.loc[1]]='[  ]'
                                minesidentified += 1
                                cellsdetected += 1
                                board[k.loc[0]][k.loc[1]].hidden=False
    
                    if (board[i][j].numneighbors-board[i][j].numminesaround)-board[i][j].numsafeidentified==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedsafe=True
                                board[k.loc[0]][k.loc[1]].hidden=False
                                gameboard[k.loc[0]][k.loc[1]]='[ S ]'
                                cellsdetected += 1
                                safefound += 1
       
        for i in range(boardlen):
            for j in range(boardlen):
        
                if cellsdetected != boardlen**2:

                        if board[i][j].hidden==True:
                            spots=0
                            neighbors=0
                            mines=0
                            cell=board[i][j]
                            for z in cell.neighbors:
                                if board[z.loc[0]][z.loc[1]].identifiedsafe==True and board[z.loc[0]][z.loc[1]].visited==True:
                                    spots += board[z.loc[0]][z.loc[1]].returnhiddenneighbors(board)
                                    mines += board[z.loc[0]][z.loc[1]].returnmineneighbors()-z.returnminesidentified(board)
                                    neighbors += 1
                            if neighbors==0:
                                board[cell.loc[0]][cell.loc[1]].p=0.5

                            if neighbors==1:
                                board[cell.loc[0]][cell.loc[1]].p=mines/spots
                            #if a cell is in the intersection of multiple cells that have been identified as safe with
                            #clues, we extend a window which opens the safe cells along with all of their neighbors
                            if neighbors>=2:
                                miny=min([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxy=max([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                minx=min([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxx=max([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                simboard=[board[b][max(0,minx-1):min(len(board),maxx+2)] for b in range(max(0,miny-1),min(len(board),maxy+2))]
                                simlst=[]
    
                                safecells=[]
                                for w in cell.neighbors:
                                    if board[w.loc[0]][w.loc[1]].identifiedsafe==True and board[w.loc[0]][w.loc[1]].visited==True:
                                        safecells.append(board[w.loc[0]][w.loc[1]])
                                acceptablescenarios=0
                                hiddencells=[]
                                for y in simboard:
                                    for x in y:
                                            if board[x.loc[0]][x.loc[1]].hidden==True:
                                                hiddencells.append(board[x.loc[0]][x.loc[1]])
                               
                                
                                #simulate every mine combination in the window using binary representation. Every
                                #number in binary number represents whether the corresponding hidden cell will be 
                                #mine or not in the simulation. 
                                for c in range(2**(len(hiddencells))):
                                    binary=bin(c)[2:].zfill(len(hiddencells))
                                    for d in range(len(binary)):
                                        if int(binary[d])==0:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=False
                                        if int(binary[d])==1:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=True
                                    #check if mine combination is possible 
                                    if isvalid(safecells,board):
                                        acceptablescenarios += 1
                                        for e in range(len(binary)):
                                            if int(binary[e])==1:
                                                board[hiddencells[e].loc[0]][hiddencells[e].loc[1]].minecount += 1
                               
                                #probability of cell being a mine is number of scenarios in which cell is a mine
                                #divided by total number of scenarios 
                                for f in [board[g][max(0,minx-1):min(len(board),maxx+2)] for g in range(max(0,miny-1),min(len(board),maxy+2))]:
                                    for h in f:
                                        if board[h.loc[0]][h.loc[1]].hidden==True:
                                            if board[h.loc[0]][h.loc[1]].minecount != 0:
                                                if board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios>board[h.loc[0]][h.loc[1]].p:
                                                    board[h.loc[0]][h.loc[1]].p=board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios
                                            else:
                                                board[h.loc[0]][h.loc[1]].p=0
                                                
                

                                            
                                            if board[h.loc[0]][h.loc[1]].p==1:
                                                board[h.loc[0]][h.loc[1]].identifiedmine=True
                                                board[h.loc[0]][h.loc[1]].hidden=False
                                                gameboard[h.loc[0]][h.loc[1]]='[  ]'
                                                cellsdetected += 1
                                                
                                            else:
                                                if board[h.loc[0]][h.loc[1]].p==0:
                                                    board[h.loc[0]][h.loc[1]].identifiedsafe=True
                                                    board[h.loc[0]][h.loc[1]].hidden=False
                                                    gameboard[h.loc[0]][h.loc[1]]='[ S ]'
                                                    cellsdetected += 1
                        
                                                board[h.loc[0]][h.loc[1]].identifiedmine=False

                                            board[h.loc[0]][h.loc[1]].minecount=0
        
                              
                        for t in range(boardlen):
                            for u in range(boardlen):
                                if board[t][u].identifiedsafe==True and board[t][u].visited==True:
                            
                                    board[t][u].numsafeidentified=board[t][u].returnsafeidentified(board)
                                    if board[t][u].numminesaround-board[t][u].returnminesidentified(board)==board[t][u].returnhiddenneighbors(board):
                                        for s in board[t][u].neighbors:
                                            if board[s.loc[0]][s.loc[1]].hidden==True:
                                                board[s.loc[0]][s.loc[1]].identifiedmine=True
                                                gameboard[s.loc[0]][s.loc[1]]='[  ]'
                                                minesidentified += 1
                                                cellsdetected += 1
                                                board[s.loc[0]][s.loc[1]].hidden=False
                                                
    
                                    if (board[t][u].numneighbors-board[t][u].numminesaround)-board[t][u].numsafeidentified==board[t][u].returnhiddenneighbors(board):
                                        for r in board[t][u].neighbors:
                                            if board[r.loc[0]][r.loc[1]].hidden==True:
                                                board[r.loc[0]][r.loc[1]].identifiedsafe=True
                                                board[r.loc[0]][r.loc[1]].hidden=False
                                                gameboard[r.loc[0]][r.loc[1]]='[ S ]'
                                                safefound += 1
                                                cellsdetected += 1
        for i in gameboard:
            print(*i)
        print('')
        if cellsdetected==len(board)**2:
            gameover=True
            
        
    return minesidentified/(minesblown+minesidentified)
#Same exact function without any printing statements. This function only outputs the score at the end of the game.
def Astarminesweeper2(board):
    gameboard=[]
    maxdigits=len(str(len(board)**2-1))
    boardlen=len(board)
    for i in range(boardlen):
        row=[]
        for j in range(boardlen):
            row.append('['+'0'*(maxdigits-len(str(i*boardlen+j)))+str(i*boardlen+j)+']')
        gameboard.append(row)
    gameover=False
    cellsdetected=0
    minesidentified=0
    minesblown=0
    safefound=0

        
    while not gameover: 
        safecell=False
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].visited==False and board[i][j].identifiedsafe==True:
                    location=board[i][j].loc
                    safecell=True
                    

        if safecell==False:
            lst=[]
            for i in range(boardlen):
                for j in range(boardlen):
                    if board[i][j].hidden==True:
                        lst.append(board[i][j]) 
            pq=Pqueue()
            pq.buildheap(lst)
            location=pq.popmin().loc
        if board[location[0]][location[1]].hidden==True:
            cellsdetected += 1
            
        board[location[0]][location[1]].hidden=False
        board[location[0]][location[1]].visited=True

        if board[location[0]][location[1]].mine==True:
            board[location[0]][location[1]].identifiedmine=True
            gameboard[location[0]][location[1]]="[  ]"
            minesblown += 1
            
            
        else:
            board[location[0]][location[1]].identifiedsafe=True

            board[location[0]][location[1]].numminesaround=board[location[0]][location[1]].returnmineneighbors()
            gameboard[location[0]][location[1]]="["+"S"+str(board[location[0]][location[1]].numminesaround)+"]"
            if board[location[0]][location[1]].hidden==True:
                safefound += 1
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].identifiedsafe==True and board[i][j].visited==True:
                    board[i][j].numsafeidentified=board[i][j].returnsafeidentified(board)
                    if board[i][j].numminesaround-board[i][j].returnminesidentified(board)==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedmine=True
                                gameboard[k.loc[0]][k.loc[1]]='[  ]'
                                minesidentified += 1
                                cellsdetected += 1
                                board[k.loc[0]][k.loc[1]].hidden=False
    
                    if (board[i][j].numneighbors-board[i][j].numminesaround)-board[i][j].numsafeidentified==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedsafe=True
                                board[k.loc[0]][k.loc[1]].hidden=False
                                gameboard[k.loc[0]][k.loc[1]]='[ S ]'
                                safefound += 1
                                cellsdetected += 1
        for i in range(boardlen):
            for j in range(boardlen):
        
                if cellsdetected != boardlen**2:

                        if board[i][j].hidden==True:
                            spots=0
                            neighbors=0
                            mines=0
                            cell=board[i][j]
                            for z in cell.neighbors:
                                if board[z.loc[0]][z.loc[1]].identifiedsafe==True and board[z.loc[0]][z.loc[1]].visited==True:
                                    spots += board[z.loc[0]][z.loc[1]].returnhiddenneighbors(board)
                                    mines += board[z.loc[0]][z.loc[1]].returnmineneighbors()-z.returnminesidentified(board)
                                    neighbors += 1
                            if neighbors==0:
                                board[cell.loc[0]][cell.loc[1]].p=0.5

                            if neighbors==1:
                                board[cell.loc[0]][cell.loc[1]].p=mines/spots
                            if neighbors>=2:
                                miny=min([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxy=max([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                minx=min([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxx=max([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                simboard=[board[b][max(0,minx-1):min(len(board),maxx+2)] for b in range(max(0,miny-1),min(len(board),maxy+2))]

    
                                safecells=[]
                                for w in cell.neighbors:
                                    if board[w.loc[0]][w.loc[1]].identifiedsafe==True and board[w.loc[0]][w.loc[1]].visited==True:
                                        safecells.append(board[w.loc[0]][w.loc[1]])
                                acceptablescenarios=0
                                hiddencells=[]
                                for y in simboard:
                                    for x in y:
                                            if board[x.loc[0]][x.loc[1]].hidden==True:
                                                hiddencells.append(board[x.loc[0]][x.loc[1]])

                    
                                        
                                for c in range(2**(len(hiddencells))):
                                    binary=bin(c)[2:].zfill(len(hiddencells))
                                    for d in range(len(binary)):
                                        if int(binary[d])==0:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=False
                                        if int(binary[d])==1:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=True
                                    if isvalid(safecells,board):
                                        acceptablescenarios += 1
                                        for e in range(len(binary)):
                                            if int(binary[e])==1:
                                                board[hiddencells[e].loc[0]][hiddencells[e].loc[1]].minecount += 1
                            
            
                                for f in [board[g][max(0,minx-1):min(len(board),maxx+2)] for g in range(max(0,miny-1),min(len(board),maxy+2))]:
                                    for h in f:
                                        if board[h.loc[0]][h.loc[1]].hidden==True:
                                            if board[h.loc[0]][h.loc[1]].minecount != 0:
                                                if board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios>board[h.loc[0]][h.loc[1]].p:
                                                    board[h.loc[0]][h.loc[1]].p=board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios
                                            else:
                                                board[h.loc[0]][h.loc[1]].p=0
                                                
                

            
                                            if board[h.loc[0]][h.loc[1]].p==1:
                                                board[h.loc[0]][h.loc[1]].identifiedmine=True
                                                board[h.loc[0]][h.loc[1]].hidden=False
                                                gameboard[h.loc[0]][h.loc[1]]='[  ]'
                                                cellsdetected += 1
        
                                            else:
                                                if board[h.loc[0]][h.loc[1]].p==0:
                                                    board[h.loc[0]][h.loc[1]].identifiedsafe=True
                                                    board[h.loc[0]][h.loc[1]].hidden=False
                                                    gameboard[h.loc[0]][h.loc[1]]='[ S ]'
                                                    cellsdetected += 1
        
                        
                                                board[h.loc[0]][h.loc[1]].identifiedmine=False

                                            board[h.loc[0]][h.loc[1]].minecount=0
        
                              
                        for t in range(boardlen):
                            for u in range(boardlen):
                                if board[t][u].identifiedsafe==True and board[t][u].visited==True:
                                    board[t][u].numsafeidentified=board[t][u].returnsafeidentified(board)
                                    if board[t][u].numminesaround-board[t][u].returnminesidentified(board)==board[t][u].returnhiddenneighbors(board):
                                        for s in board[t][u].neighbors:
                                            if board[s.loc[0]][s.loc[1]].hidden==True:
                                                board[s.loc[0]][s.loc[1]].identifiedmine=True
                                                gameboard[s.loc[0]][s.loc[1]]='[  ]'
                                                minesidentified += 1
                                                cellsdetected += 1
                                                board[s.loc[0]][s.loc[1]].hidden=False
    
                                    if (board[t][u].numneighbors-board[t][u].numminesaround)-board[t][u].numsafeidentified==board[t][u].returnhiddenneighbors(board):
                                        for r in board[t][u].neighbors:
                                            if board[r.loc[0]][r.loc[1]].hidden==True:
                                                board[r.loc[0]][r.loc[1]].identifiedsafe=True
                                                board[r.loc[0]][r.loc[1]].hidden=False
                                                gameboard[r.loc[0]][r.loc[1]]='[ S ]'
                                                safefound += 1
                                                cellsdetected += 1
        
        if cellsdetected==boardlen**2:
            gameover=True
            
        
    return minesidentified/(minesblown+minesidentified)
#Combines the best strategies from the baseline agent and the probabilistic agent. Probability is only used to 
#determine if a cell is definitely a mine or safe. Otherwise, the same sequential pattern is followed from the 
#baseline agent to visit hidden cells. This function must perform equal to or better than the baseline agent. 
def Astarminesweeper3(board):
    gameboard=[]
    maxdigits=len(str(len(board)**2-1))
    boardlen=len(board)
    for i in range(boardlen):
        row=[]
        for j in range(boardlen):
            row.append('['+'0'*(maxdigits-len(str(i*boardlen+j)))+str(i*boardlen+j)+']')
        gameboard.append(row)
    gameover=False
    cellsdetected=0
    minesidentified=0
    minesblown=0
    safefound=0

        
    while not gameover: 
        safecell=False
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].visited==False and board[i][j].identifiedsafe==True:
                    location=board[i][j].loc
                    safecell=True
                    

        if safecell==False:
            for i in range(boardlen):
                for j in range(boardlen):
                    if board[i][j].visited==False and board[i][j].identifiedmine==False:
                        location=(i,j) 
        
        if board[location[0]][location[1]].hidden==True:
            cellsdetected += 1
            
        board[location[0]][location[1]].hidden=False
        board[location[0]][location[1]].visited=True

        if board[location[0]][location[1]].mine==True:
            board[location[0]][location[1]].identifiedmine=True
            gameboard[location[0]][location[1]]="[  ]"
            minesblown += 1
            
            
        else:
            board[location[0]][location[1]].identifiedsafe=True

            board[location[0]][location[1]].numminesaround=board[location[0]][location[1]].returnmineneighbors()
            gameboard[location[0]][location[1]]="["+"S"+str(board[location[0]][location[1]].numminesaround)+"]"
            if board[location[0]][location[1]].hidden==True:
                safefound += 1
        for i in range(boardlen):
            for j in range(boardlen):
                if board[i][j].identifiedsafe==True and board[i][j].visited==True:
                    board[i][j].numsafeidentified=board[i][j].returnsafeidentified(board)
                    if board[i][j].numminesaround-board[i][j].returnminesidentified(board)==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedmine=True
                                gameboard[k.loc[0]][k.loc[1]]='[  ]'
                                minesidentified += 1
                                cellsdetected += 1
                                board[k.loc[0]][k.loc[1]].hidden=False
    
                    if (board[i][j].numneighbors-board[i][j].numminesaround)-board[i][j].numsafeidentified==board[i][j].returnhiddenneighbors(board):
                        for k in board[i][j].neighbors:
                            if board[k.loc[0]][k.loc[1]].hidden==True:
                                board[k.loc[0]][k.loc[1]].identifiedsafe=True
                                board[k.loc[0]][k.loc[1]].hidden=False
                                gameboard[k.loc[0]][k.loc[1]]='[ S ]'
                                safefound += 1
                                cellsdetected += 1
        for i in range(boardlen):
            for j in range(boardlen):
        
                if cellsdetected != boardlen**2:

                        if board[i][j].hidden==True:
                            spots=0
                            neighbors=0
                            mines=0
                            cell=board[i][j]
                            for z in cell.neighbors:
                                if board[z.loc[0]][z.loc[1]].identifiedsafe==True and board[z.loc[0]][z.loc[1]].visited==True:
                                    spots += board[z.loc[0]][z.loc[1]].returnhiddenneighbors(board)
                                    mines += board[z.loc[0]][z.loc[1]].returnmineneighbors()-z.returnminesidentified(board)
                                    neighbors += 1
                            if neighbors==0:
                                board[cell.loc[0]][cell.loc[1]].p=0.5

                            if neighbors==1:
                                board[cell.loc[0]][cell.loc[1]].p=mines/spots
                            if neighbors>=2:
                                miny=min([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxy=max([a.loc[0] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                minx=min([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                maxx=max([a.loc[1] for a in cell.neighbors if board[a.loc[0]][a.loc[1]].identifiedsafe==True and board[a.loc[0]][a.loc[1]].visited==True])
                                simboard=[board[b][max(0,minx-1):min(len(board),maxx+2)] for b in range(max(0,miny-1),min(len(board),maxy+2))]

    
                                safecells=[]
                                for w in cell.neighbors:
                                    if board[w.loc[0]][w.loc[1]].identifiedsafe==True and board[w.loc[0]][w.loc[1]].visited==True:
                                        safecells.append(board[w.loc[0]][w.loc[1]])
                                acceptablescenarios=0
                                hiddencells=[]
                                for y in simboard:
                                    for x in y:
                                            if board[x.loc[0]][x.loc[1]].hidden==True:
                                                hiddencells.append(board[x.loc[0]][x.loc[1]])

                    
                                        
                                for c in range(2**(len(hiddencells))):
                                    binary=bin(c)[2:].zfill(len(hiddencells))
                                    for d in range(len(binary)):
                                        if int(binary[d])==0:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=False
                                        if int(binary[d])==1:
                                            board[hiddencells[d].loc[0]][hiddencells[d].loc[1]].identifiedmine=True
                                    if isvalid(safecells,board):
                                        acceptablescenarios += 1
                                        for e in range(len(binary)):
                                            if int(binary[e])==1:
                                                board[hiddencells[e].loc[0]][hiddencells[e].loc[1]].minecount += 1
                            
            
                                for f in [board[g][max(0,minx-1):min(len(board),maxx+2)] for g in range(max(0,miny-1),min(len(board),maxy+2))]:
                                    for h in f:
                                        if board[h.loc[0]][h.loc[1]].hidden==True:
                                            if board[h.loc[0]][h.loc[1]].minecount != 0:
                                                if board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios>board[h.loc[0]][h.loc[1]].p:
                                                    board[h.loc[0]][h.loc[1]].p=board[h.loc[0]][h.loc[1]].minecount/acceptablescenarios
                                            else:
                                                board[h.loc[0]][h.loc[1]].p=0
                                                
                

            
                                            if board[h.loc[0]][h.loc[1]].p==1:
                                                board[h.loc[0]][h.loc[1]].identifiedmine=True
                                                board[h.loc[0]][h.loc[1]].hidden=False
                                                gameboard[h.loc[0]][h.loc[1]]='[  ]'
                                                cellsdetected += 1
        
                                            else:
                                                if board[h.loc[0]][h.loc[1]].p==0:
                                                    board[h.loc[0]][h.loc[1]].identifiedsafe=True
                                                    board[h.loc[0]][h.loc[1]].hidden=False
                                                    gameboard[h.loc[0]][h.loc[1]]='[ S ]'
                                                    cellsdetected += 1
        
                        
                                                board[h.loc[0]][h.loc[1]].identifiedmine=False

                                            board[h.loc[0]][h.loc[1]].minecount=0
        
                              
                        for t in range(boardlen):
                            for u in range(boardlen):
                                if board[t][u].identifiedsafe==True and board[t][u].visited==True:
                                    board[t][u].numsafeidentified=board[t][u].returnsafeidentified(board)
                                    if board[t][u].numminesaround-board[t][u].returnminesidentified(board)==board[t][u].returnhiddenneighbors(board):
                                        for s in board[t][u].neighbors:
                                            if board[s.loc[0]][s.loc[1]].hidden==True:
                                                board[s.loc[0]][s.loc[1]].identifiedmine=True
                                                gameboard[s.loc[0]][s.loc[1]]='[  ]'
                                                minesidentified += 1
                                                cellsdetected += 1
                                                board[s.loc[0]][s.loc[1]].hidden=False
    
                                    if (board[t][u].numneighbors-board[t][u].numminesaround)-board[t][u].numsafeidentified==board[t][u].returnhiddenneighbors(board):
                                        for r in board[t][u].neighbors:
                                            if board[r.loc[0]][r.loc[1]].hidden==True:
                                                board[r.loc[0]][r.loc[1]].identifiedsafe=True
                                                board[r.loc[0]][r.loc[1]].hidden=False
                                                gameboard[r.loc[0]][r.loc[1]]='[ S ]'
                                                safefound += 1
                                                cellsdetected += 1
        
        if cellsdetected==boardlen**2:
            gameover=True
            
        
    return minesidentified/(minesblown+minesidentified)

#Plotting function that plots score of minesweeper game versus number of mines on the board. Function can be edited
#to change board size,range of number of mines,number of iterations, and algorithm to use. 
def densityplotter():
    successrates=[]
    densityvalues=[]
    successrates2=[]
    for d in range(1,49,5):
        densityvalues.append(d)
        successrate=[]
        successrate2=[]
        for i in range(80):
            a=createboard(7,d)
            successrate2.append(Astarminesweeper3(a))
            boardreset(a)
            successrate.append(minesweeper(a))
        successrates.append(sum(successrate)/len(successrate))
        successrates2.append(sum(successrate2)/len(successrate2))
    plt.plot(densityvalues,successrates)
    plt.plot(densityvalues,successrates2)
    plt.show()
#validation checker that determines if a given mine combination is possbile based on safe cell clues
def isvalid(safelst,board):
    for i in safelst:
        if board[i.loc[0]][i.loc[1]].numminesaround != board[i.loc[0]][i.loc[1]].returnminesidentified(board):
            return False      
    return True
            
#prints representation of board with apple sign being a mine and number indicating safe cell and number of mine neighbors          
def printboard(board):
    gameboard=[]
    for i in board:
        row=[]
        for j in i:
            if j.mine==True:
                row.append('[  ]')
            else:
                row.append('[ '+str(j.returnmineneighbors())+' ]')
        gameboard.append(row)
    for i in gameboard:
        print(*i)
    print('')
#resets attributes for cells in the board 
def boardreset(board):
    for i in range(len(board)):
        for j in range(len(board)):
            board[i][j].hidden=True
            board[i][j].identifiedsafe=False
            board[i][j].identifiedmine=False
            board[i][j].visited=False
            board[i][j].p=0
            board[i][j].hiddenneighbors=0
            board[i][j].numminesidentified=0
            board[i][j].numsafeidentified=0
            board[i][j].numminesaround=0
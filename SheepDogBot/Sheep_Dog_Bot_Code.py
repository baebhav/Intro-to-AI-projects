import matplotlib.pyplot as plt
import math
import random
#intialize the cell objects which make up the board
#create connections to neighboring cells that will be updated
#include state attributes such as the object occupying it 
class GridCell(object):
    def __init__(self,loc):
        self.loc=loc
        self.state='empty'
        self.top=None
        self.right=None
        self.bottom=None
        self.left=None
        self.bottomtrapped=False
        self.righttrapped=False
        self.neighbors=[]
#build the board
def buildgrid():
    grid=[]
    for i in range(8):
        row=[]
        for j in range(8):
            row.append(GridCell((i,j)))
        grid.append(row)
    return grid
#euclidean function which determines how far a bot is from the sheep
def euclidean(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
#update the connections between all the cells and intialize the two bots and sheep on the board
#the information from this function is input into the sheepbot simulation function
def initialize(grid):
    for i in range(8):
        for j in range(8):
            if i-1>=0:
                grid[i][j].top=(i-1,j)
                grid[i][j].neighbors.append((i-1,j))
            if i+1<=7:
                grid[i][j].bottom=(i+1,j)
                grid[i][j].neighbors.append((i+1,j))
            if j-1>=0:
                grid[i][j].left=(i,j-1)
                grid[i][j].neighbors.append((i,j-1))
            if j+1<=7:
                grid[i][j].right=(i,j+1)
                grid[i][j].neighbors.append((i,j+1))
            
    sheepspot=random.randint(0,63)
    grid[sheepspot//8][sheepspot%8].state='sheep'
    bot1spots=[i for i in range(64) if i != sheepspot]
    bot1spot=bot1spots[random.randint(0,62)]
    grid[bot1spot//8][bot1spot%8].state='bot1'
    bot2spots=[i for i in range(64) if (i != sheepspot and i!= bot1spot)]
    bot2spot=bot2spots[random.randint(0,61)]
    grid[bot2spot//8][bot2spot%8].state='bot2'
    return [grid,(sheepspot//8,sheepspot%8),(bot1spot//8,bot1spot%8),(bot2spot//8,bot2spot%8)]
def sheepbotcompetition(gridinfo):
    gameover=False
    sheepspot=gridinfo[1]
    bot1spot=gridinfo[2]
    bot2spot=gridinfo[3]
    rounds=0
    grid=gridinfo[0]
    physicalgrid=[]
    sheepmove=None
    bot1hold=False
    bot2hold=False
    randomrestart=False
    while gameover==False:
        #print the board at the start of every round
        
        for i in range(len(grid)):
            row=[]
            for j in range(len(grid)):
                if grid[i][j].state=='empty':
                    row.append('[ ]')
                if grid[i][j].state=='bot1':
                    row.append('[D1]')
                if grid[i][j].state=='bot2':
                    row.append('[D2]')
                if grid[i][j].state=='sheep':
                    row.append('[S]')
            print(*row)
        #if random restart is required, randomly move the two bots to neighbor cells
        if randomrestart==True:
            bot1neighborlst=[i for i in grid[bot1spot[0]][bot1spot[1]].neighbors if grid[i[0]][i[1]].state=='empty']
            bot2neighborlst=[i for i in grid[bot2spot[0]][bot2spot[1]].neighbors if grid[i[0]][i[1]].state=='empty']
            newbot1spot=bot1neighborlst[random.randint(0,len(bot1neighborlst)-1)]
            newbot2spot=bot2neighborlst[random.randint(0,len(bot2neighborlst)-1)]
            if grid[newbot1spot[0]][newbot1spot[1]].state=='empty':
                grid[bot1spot[0]][bot1spot[1]].state='empty'
                bot1spot=newbot1spot
                grid[bot1spot[0]][bot1spot[1]].state='bot1'
            if grid[newbot2spot[0]][newbot2spot[1]].state=='empty':
                grid[bot2spot[0]][bot2spot[1]].state='empty'
                bot2spot=newbot2spot
                grid[bot2spot[0]][bot2spot[1]].state='bot2'
            
            
        else:
            #if bot1 hasn't recieved instructions to hold its position, continue
            if bot1hold==False:
                #if bot1 has occupied a position that traps the sheep, just follow the last movement that the sheep made
                if (grid[bot1spot[0]][bot1spot[1]].righttrapped or grid[bot1spot[0]][bot1spot[1]].bottomtrapped):
    
                        
                    if sheepmove=='up':
                        if bot1spot[0]-1>=0 and grid[bot1spot[0]-1][bot1spot[1]].state=='empty':
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            bot1spot=(bot1spot[0]-1,bot1spot[1])
                            grid[bot1spot[0]][bot1spot[1]].state='bot1'
                     
                    if sheepmove=='down':
                        if bot1spot[0]+1<=7 and grid[bot1spot[0]+1][bot1spot[1]].state=='empty':
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            bot1spot=(bot1spot[0]+1,bot1spot[1])
                            grid[bot1spot[0]][bot1spot[1]].state='bot1'
             
                    if sheepmove=='left':
                        if bot1spot[1]-1>=0 and grid[bot1spot[0]][bot1spot[1]-1].state=='empty':
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            bot1spot=(bot1spot[0],bot1spot[1]-1)
                            grid[bot1spot[0]][bot1spot[1]].state='bot1'
                     
                    if sheepmove=='right':
                        if bot1spot[1]+1<=7 and grid[bot1spot[0]][bot1spot[1]+1].state=='empty':
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            bot1spot=(bot1spot[0],bot1spot[1]+1)
                            grid[bot1spot[0]][bot1spot[1]].state='bot1'
                        
                else:
                    #if not, look for a cell which is a bottom or right neighbor to the sheep cell. If no such cell
                    #exists, search for a neigboring cell that is closest to the sheep cell
                    closestcell1=grid[bot1spot[0]][bot1spot[1]].neighbors[0]
                    mindistance1=euclidean(grid[bot1spot[0]][bot1spot[1]].neighbors[0],sheepspot)
                    topneighbor=False
                    leftneighbor=False
                    toploc=None
                    leftloc=None
                    trapped=False
                    for i in grid[bot1spot[0]][bot1spot[1]].neighbors:
                        if grid[i[0]][i[1]].state=='empty':
                            top=grid[i[0]][i[1]].top
                            left=grid[i[0]][i[1]].left
                            if top and grid[top[0]][top[1]].state=='sheep':
                                trapped=True
                                topneighbor=True
                                toploc=i
                                
                            if left and grid[left[0]][left[1]].state=='sheep':
                                trapped=True
                                leftneighbor=True
                                leftloc=i
                            if euclidean(i,sheepspot)<=mindistance1:
                                mindistance1=euclidean(i,sheepspot)
                                closestcell1=i
                    if topneighbor and leftneighbor:
                        if (bot1spot[0]-bot2spot[0]>0 or bot1spot[1]-bot2spot[1]<0):
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            grid[toploc[0]][toploc[1]].state='bot1'
                            grid[toploc[0]][toploc[1]].bottomtrapped=True
                            bot1spot=(toploc[0],toploc[1])
                        else:
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            grid[leftloc[0]][leftloc[1]].state='bot1'
                            grid[leftloc[0]][leftloc[1]].righttrapped=True
                            bot1spot=(leftloc[0],leftloc[1])
                            
                    else:
                        if topneighbor:
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            grid[toploc[0]][toploc[1]].state='bot1'
                            grid[toploc[0]][toploc[1]].bottomtrapped=True
                            bot1spot=(toploc[0],toploc[1])
                        if leftneighbor:
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            grid[leftloc[0]][leftloc[1]].state='bot1'
                            grid[leftloc[0]][leftloc[1]].righttrapped=True
                            bot1spot=(leftloc[0],leftloc[1])
            
                    if trapped==False:
                        if grid[closestcell1[0]][closestcell1[1]].state=='empty' and mindistance1<euclidean(bot1spot,sheepspot):
                            grid[bot1spot[0]][bot1spot[1]].state='empty'
                            bot1spot=closestcell1
                            grid[bot1spot[0]][bot1spot[1]].state='bot1'
            #same exact instructions, except it's for bot2                
            if bot2hold==False:
                if (grid[bot2spot[0]][bot2spot[1]].righttrapped or grid[bot2spot[0]][bot2spot[1]].bottomtrapped):
    
                    
                    if sheepmove=='up':
                        if bot2spot[0]-1>=0 and grid[bot2spot[0]-1][bot2spot[1]].state=='empty':
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            bot2spot=(bot2spot[0]-1,bot2spot[1])
                            grid[bot2spot[0]][bot2spot[1]].state='bot2'
                          
                    if sheepmove=='down':
                        if bot2spot[0]+1<=7 and grid[bot2spot[0]+1][bot2spot[1]].state=='empty':
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            bot2spot=(bot2spot[0]+1,bot2spot[1])
                            grid[bot2spot[0]][bot2spot[1]].state='bot2'
                        
                    if sheepmove=='left':
                        if bot2spot[1]-1>=0 and grid[bot2spot[0]][bot2spot[1]-1].state=='empty':
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            bot2spot=(bot2spot[0],bot2spot[1]-1)
                            grid[bot2spot[0]][bot2spot[1]].state='bot2'
                         
                    if sheepmove=='right':
                        if bot2spot[1]+1<=7 and grid[bot2spot[0]][bot2spot[1]+1].state=='empty':
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            grid[bot2spot[0]][bot2spot[1]+1].state='bot2'
                            bot2spot=(bot2spot[0],bot2spot[1]+1)
             
                else:
                    closestcell2=grid[bot2spot[0]][bot2spot[1]].neighbors[0]
                    mindistance2=euclidean(grid[bot2spot[0]][bot2spot[1]].neighbors[0],sheepspot)
                    leftneighbor=False
                    topneighbor=False
                    leftloc=None
                    toploc=None
                    trapped=False
                    for j in grid[bot2spot[0]][bot2spot[1]].neighbors:
                        if grid[j[0]][j[1]].state=='empty':
                            top=grid[j[0]][j[1]].top
                            left=grid[j[0]][j[1]].left
                            if top and grid[top[0]][top[1]].state=='sheep':
                                trapped=True
                                topneighbor=True
                                toploc=j
                            if left and grid[left[0]][left[1]].state=='sheep':
                                trapped=True
                                leftneighbor=True
                                leftloc=j
                            if euclidean(j,sheepspot)<=mindistance2:
                                mindistance2=euclidean(j,sheepspot)
                                closestcell2=j
                    if topneighbor and leftneighbor:
                        if (bot2spot[0]-bot1spot[0]>0 or bot2spot[1]-bot1spot[1]<0):
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            grid[toploc[0]][toploc[1]].state='bot2'
                            grid[toploc[0]][toploc[1]].bottomtrapped=True
                            bot2spot=(toploc[0],toploc[1])
                        else:
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            grid[leftloc[0]][leftloc[1]].state='bot2'
                            grid[leftloc[0]][leftloc[1]].righttrapped=True
                            bot2spot=(leftloc[0],leftloc[1])
                    
                    else:
                        if topneighbor:
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            grid[toploc[0]][toploc[1]].state='bot2'
                            grid[toploc[0]][toploc[1]].bottomtrapped=True
                            bot2spot=(toploc[0],toploc[1])
                        if leftneighbor:
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                            grid[leftloc[0]][leftloc[1]].state='bot2'
                            grid[leftloc[0]][leftloc[1]].righttrapped=True
                            bot2spot=(leftloc[0],leftloc[1])
                            
                            
                        
                    if trapped==False:
                        if grid[closestcell2[0]][closestcell2[1]].state=='empty' and mindistance2<euclidean(bot2spot,sheepspot):
                            grid[bot2spot[0]][bot2spot[1]].state='empty'
                        
                            bot2spot=closestcell2
                       
                            grid[bot2spot[0]][bot2spot[1]].state='bot2'
        #sheep randomly moves to one its empty neighbors
        randomrestart=False
        sheepneighborlst=[i for i in grid[sheepspot[0]][sheepspot[1]].neighbors if grid[i[0]][i[1]].state=='empty']
        if len(sheepneighborlst)==0:
            randomrestart=True
        if randomrestart==False:
            grid[sheepspot[0]][sheepspot[1]].state='empty'
            
            newsheepspot=sheepneighborlst[random.randint(0,len(sheepneighborlst)-1)]
            #track the movement of the sheep
            if newsheepspot==grid[sheepspot[0]][sheepspot[1]].top:
                sheepmove='up'
            
            if newsheepspot==grid[sheepspot[0]][sheepspot[1]].left:
                sheepmove='left'
            
            if newsheepspot==grid[sheepspot[0]][sheepspot[1]].bottom:
                sheepmove='down'
            
            if newsheepspot==grid[sheepspot[0]][sheepspot[1]].right:
                sheepmove='right'
            sheepspot=newsheepspot
            
            grid[sheepspot[0]][sheepspot[1]].state='sheep'
            right=grid[sheepspot[0]][sheepspot[1]].right
            bottom=grid[sheepspot[0]][sheepspot[1]].bottom
            bot1hold=False
            bot2hold=False
            #if the sheep enters a cell which is the left neighbor or top neighbor of one of the 
            #bot cells, the sheep has trapped itself. The bot(s) will now be instructed to hold its/their
            #position
            if right and grid[right[0]][right[1]].state=='bot1':
                bot1hold=True
                grid[bot1spot[0]][bot1spot[1]].righttrapped=True
                
            if bottom and grid[bottom[0]][bottom[1]].state=='bot1':
                grid[bot1spot[0]][bot1spot[1]].bottomtrapped=True
                
                bot1hold=True
            if right and grid[right[0]][right[1]].state=='bot2':
                grid[bot2spot[0]][bot2spot[1]].righttrapped=True
                
                bot2hold=True
            if bottom and grid[bottom[0]][bottom[1]].state=='bot2':
                grid[bot2spot[0]][bot2spot[1]].bottomtrapped=True
                
                bot2hold=True
        #end the simulation and return the number of rounds if the sheep is pinned to the top left corner
        if grid[0][0].state=='sheep' and (grid[0][1].state=='bot1' or grid[0][1].state=='bot2') and (grid[1][0].state=='bot2' or grid[1][0].state=='bot1'):
            return rounds
        rounds += 1
        
        
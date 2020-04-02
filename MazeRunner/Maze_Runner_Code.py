#imports libraries for calculating euclidean values, plotting, BFS queue, and probability generation respectively
import math
import matplotlib.pyplot as plt
from collections import deque
import random
import time
#Creates object for use during genetic algorithm building. It's main advantage
#is it allows for sorting of mazes based on the path attribute which can be initialized
#to represent whatever metric we're using for the genetic algorithm such as max fringe
#size or shortest path
class Maze(object):
    def __init__(self,maze,path):
        self.maze=maze
        self.path=path
#Creates priority queue class that will be used for the A* algorithms. The methods are
# the same for a regular priority queue but has been modified to arange 
#elements according to their corresponding f(n) value attributes. 
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
            if self.heap[index].f<self.heap[index//2].f:
                tmp=self.heap[index//2]
                self.heap[index//2]=self.heap[index]
                self.heap[index]=tmp
            index=index//2


    def movedown(self,i):
        while (2*i)<=self.size:
            minindex=self.minchild(i)
            if self.heap[i].f>self.heap[minindex].f:
                parent=self.heap[i]
                self.heap[i]=self.heap[minindex]
                self.heap[minindex]=parent
            i=minindex

    def minchild(self,index):
        if 2*index+1>self.size:
            return 2*index
        else:
            if self.heap[2*index].f<self.heap[2*index+1].f:
                return 2*index
            else:
                return 2*index+1


#Creates the maze cell object that will be used during maze 
#generation and maze traversal. Each cell is either intialized as
#empty or filled and has pointers to the right,left,top,and bottom
#cells. They also have additional attributes such as if they have been
#visited or their parent cell that helps during traversal algorithms
class Cell(object):             
    def __init__(self,Empty):     
        if Empty==1:
            self.isEmpty=True  
        else:
            self.isEmpty=False
        self.euclidean=None
        self.distance=None
        self.f=10000
        self.start=False
        self.goal=False
        self.top=None
        self.bottom=None
        self.left=None
        self.right=None
        self.parent=None
        self.visited=False
        self.manhatten=None
        self.fire=False
        self.loc=(None,None)
#generates a list of lists that contains the cell objects. The cells
#are initialized based on the the probability p of a cell being 
#blocked using a random number generator. This p only applies to cells
#that are not the start or goal because they need to be open. 
def generatemaze(dim,p):
    maze=[]
    for i in range(dim):
        row=[]
        for j in range(dim):
            num=random.randint(0,99)
            if (i,j) == (0,0) or (i,j) == (dim-1,dim-1):
                cell=Cell(1)
            else:
                if num<=(p*100)-1:
                    cell=Cell(0)
                else:
                    cell=Cell(1)
            row.append(cell)
        maze.append(row)
    return maze
#Based on the cells location in the maze, its pointers to neighboring
#cells are updated. The top left cell is then initialized as the start
#cell and the bottom right cell is intialized as the goal cell. We also
#initialize the cells location within the cell, its euclidean value, and its
#manhatten value. 
def createpointer(maze):
    for i in range(len(maze)):
        for j in range(len(maze)):
            maze[i][j].euclidean=(((j-len(maze)+1)**2)+((i-len(maze)+1)**2))**(0.5)
            maze[i][j].manhatten=abs(i-len(maze)+1)+abs(j-len(maze)+1)
            maze[i][j].loc=(i,j)
            
            if maze[i][j].isEmpty==True:
                if j-1>=0:
                    if maze[i][j-1].isEmpty==True:
                        maze[i][j].left=maze[i][j-1]
                if j+1<=len(maze)-1:
                    if maze[i][j+1].isEmpty==True:
                        maze[i][j].right=maze[i][j+1]
                if i-1>=0:
                    if maze[i-1][j].isEmpty==True:
                        maze[i][j].top=maze[i-1][j]
                if i+1<=len(maze)-1:
                    if maze[i+1][j].isEmpty==True:
                        maze[i][j].bottom=maze[i+1][j]
    maze[0][0].start=True
    maze[len(maze)-1][len(maze)-1].goal=True
    
    
    return maze



    
#Standard BFS algorithm using a queue. The queue starts off only
#containing the start cell and then continues to add neighboring
#cells to the queue until the goal cell is found. We always pop off
#cells from the start of the queue and add their neighbors
#to the end of the queue. Throughtout this process cells are marked as 
#visited so we don't revisit them and their parent pointers are updated so 
#we can return a path from start to goal if a path exists

def BFS(maze):
            
    queue=deque()
    queue.append(maze[0][0])
    maze[0][0].visited=True
    cells=[]
    blocked=[]
    for i in maze:
        for j in i:
            if j.isEmpty==False:
                blocked.append(j.loc)
    while len(queue)>0:
        node=queue.popleft()
        if node.goal==True:
            while node.start != True:
                cells.append(node.loc)
                node=node.parent
            cells.append((0,0))
            return cells[::-1]


        if node.top and not node.top.visited:
            queue.append(node.top)
            node.top.parent=node
            node.top.visited=True
        if node.left and not node.left.visited:
            queue.append(node.left)
            node.left.parent=node
            node.left.visited=True
        if node.bottom and not node.bottom.visited:
            queue.append(node.bottom)
            node.bottom.parent=node
            node.bottom.visited=True
        if node.right and not node.right.visited:
            queue.append(node.right)
            node.right.parent=node
            node.right.visited=True
    
    
    return 'Failure'

    
#Same exact algorithm and code as a BFS search except we are using a 
#stack instead of a queue. We want to add the bottom and right cells 
#to the stack last because we want to always move right and then down
#by default because that will most likely result in the least amount
#of iterations of the search
def DFS(maze):
    stack=[maze[0][0]]
    maze[0][0].visited=True
    cells=[]
    maxfringe=1
    while len(stack)>0:
        if len(stack)>maxfringe:
            maxfringe=len(stack)
        node=stack.pop()
        if node.goal==True:
            while node.start != True:
                cells.append(node.loc)
                node=node.parent
            cells.append((0,0))
            return maxfringe


        if node.top and not node.top.visited:
            stack.append(node.top)
            node.top.parent=node
            node.top.visited=True
        if node.left and not node.left.visited:
            stack.append(node.left)
            node.left.parent=node
            node.left.visited=True
        if node.bottom and not node.bottom.visited:
            stack.append(node.bottom)
            node.bottom.parent=node
            node.bottom.visited=True
        if node.right and not node.right.visited:
            stack.append(node.right)
            node.right.parent=node
            node.right.visited=True
    
    return 'Failure'
#function designed to show that the approach from the above DFS algorithm
#will outperform DFS algorithms that move away from the goal
def DFS2(maze):
    stack=[maze[0][0]]
    maze[0][0].visited=True
    cells=[]
    maxfringe=1
    while len(stack)>0:
        if len(stack)>maxfringe:
            maxfringe=len(stack)
        node=stack.pop()
        if node.goal==True:
            while node.start != True:
                cells.append(node.loc)
                node=node.parent
            cells.append((0,0))
            return maxfringe

        if node.bottom and not node.bottom.visited:
            stack.append(node.bottom)
            node.bottom.parent=node
            node.bottom.visited=True
        if node.right and not node.right.visited:
            stack.append(node.right)
            node.right.parent=node
            node.right.visited=True
        if node.top and not node.top.visited:
            stack.append(node.top)
            node.top.parent=node
            node.top.visited=True
        if node.left and not node.left.visited:
            stack.append(node.left)
            node.left.parent=node
            node.left.visited=True
    
    return 'Failure'


#function that plots success rate of algorithms given a range of p values. Can be
#edited to include any algorithm.
def densityplotter(dim):
    pvalues=[]
    successrate=[]
    for i in range(0,101,5):
        pvalues.append(i/100)
        successes=0
        for j in range(100):
            gra=createpointer(generatemaze(dim,(i/100)))
            sol=DFS(gra)
            if sol != 'Failure':
                successes += 1

            
        successrate.append(successes/100)
                
    plt.plot(pvalues,successrate)

    plt.xlabel('p values')
    plt.ylabel('A* Manhattan success rate')
    plt.show()
def dimplotter():
    dimvalues=[]
    completiontimes=[]
    for i in range(1,101):
        dimvalues.append(i)
        completiontime=[]
        for j in range(100):
            gra=createpointer(generatemaze(i,0.3))
            sol=Amanhatten(gra)
            time1=time.time()
            Amanhatten(gra)
            time2=time.time()-time1
            if sol != 'Failure':
                completiontime.append(time2)
        completiontimes.append(sum(completiontime)/len(completiontime))
    plt.plot(dimvalues,completiontimes)
    plt.xlabel('dim values')
    plt.ylael('average completion time for p=0.3')
    plt.show()
            
#can be edited to find average completion time for any algorithm         
def timeplotter(dim):
    pvalues=[]
    BFStimes=[]
    DFStimes=[]
    Amantimes=[]
    Aeuclidtimes=[]
    for i in range(0,41,5):
        pvalues.append(i)
        BFStime=[]
        DFStime=[]
        Amantime=[]
        Aeuclidtime=[]
        
        for j in range(100):
            gra=createpointer(generatemaze(dim,(i/100)))
            sol=DFS(gra)
            time1=time.time()
            DFS(gra)
            time2=time.time()-time1

            if sol != 'Failure':
                DFStime.append(time2)
            
        DFStimes.append((sum(DFStime))/(len(DFStime)))
                
    plt.plot(pvalues,DFStimes)
    plt.xlabel('p values')
    plt.ylabel('average time to execute DFS')
    plt.show()
#Algorithm for A* search using euclidean distance as the heuristic. As we traverse through the maze, the 
#cell distance/g(n) attribute is dynamically initialized and then added with the predetermined euclidean
#/h(n) attribute to create the f(n) attribute. Since we are using a priority queue, cells in the queue
#are ranked according to this f value.
def Aeuclid(maze):
    maze[0][0].distance=0
    maze[0][0].f=maze[0][0].euclidean
    pq=Pqueue()
    pq.buildheap([maze[0][0]])
    cells=[]
    blocked=[]
    counter=0
    while pq.size>0:
        node=pq.popmin()
        counter += 1
        if node.goal==True:
            #stores the cell locations of the path found
            while node.start != True:
                cells.append(node.loc)
                node=node.parent
            cells.append((0,0))
            return counter
        #only visit a node if its current f(n) value is less than its previous
        #f(n) value if already visited
        if node.right and node.distance+1+node.right.euclidean<node.right.f:
            node.right.distance=node.distance+1
            node.right.f=node.right.euclidean+(node.distance+1)
            pq.insert(node.right)
            node.right.parent=node
        if node.bottom and node.distance+1+node.bottom.euclidean<node.bottom.f:
            node.bottom.distance=node.distance+1
            node.bottom.f=node.bottom.euclidean+(node.distance+1)
            pq.insert(node.bottom)
            node.bottom.parent=node
        if node.left and node.distance+1+node.left.euclidean<node.left.f:
            node.left.distance=node.distance+1
            node.left.f=node.left.euclidean+(node.distance+1)
            pq.insert(node.left)
            node.left.parent=node
        if node.top and node.distance+1+node.top.euclidean<node.top.f:
            node.top.distance=node.distance+1
            node.top.f=node.top.euclidean+(node.distance+1)
            pq.insert(node.top)
            node.top.parent=node
    
    return 'Failure'
#Same exact code as A* euclidean except the heuristic is now the manhatten distance from cell to goal
def Amanhatten(maze):
    maze[0][0].distance=0
    maze[0][0].f=maze[0][0].manhatten
    pq=Pqueue()
    pq.buildheap([maze[0][0]])
    cells=[]
    blocked=[]
    counter=0
    while pq.size>0:
        node=pq.popmin()
        counter += 1
        if node.goal==True:
            while node.start != True:
                cells.append(node.loc)
                node=node.parent
            cells.append((0,0))
            return counter
            
        if node.top and node.distance+1+node.top.manhatten<node.top.f:
            node.top.distance=node.distance+1
            node.top.f=node.top.manhatten+(node.distance+1)
            pq.insert(node.top)
            node.top.parent=node
            node.top.visited=True
        if node.left and node.distance+1+node.left.manhatten<node.left.f:
            node.left.distance=node.distance+1
            node.left.f=node.left.manhatten+(node.distance+1)
            pq.insert(node.left)
            node.left.parent=node
            node.left.visited=True
        if node.bottom and node.distance+1+node.bottom.manhatten<node.bottom.f:
            node.bottom.distance=node.distance+1
            node.bottom.f=node.bottom.manhatten+(node.distance+1)
            pq.insert(node.bottom)
            node.bottom.parent=node
            node.bottom.visited=True
        if node.right and node.distance+1+node.right.manhatten<node.right.f:
            node.right.distance=node.distance+1
            node.right.f=node.right.manhatten+(node.distance+1)
            pq.insert(node.right)
            node.right.parent=node
            node.right.visited=True
    
    return 'Failure'



#maze reset function that comes in handy during 8 queens genetic algorithm. Since we are testing
#mazes using search functions, we want to reset the cell attributes that are changed after completion
#of the searches, because the children of mazes will contain the same changed attributes which will be a 
#problem when we run multiple search algorithms in one function. 
def mazereset(maze):
    for i in maze:
        for j in i:
            j.visited=False
            j.parent=None
            j.distance=None
            j.f=10000
#Application of the 8 queens genetic search algorithm. We start off with n solvable mazes that represent the 
#first generation of the population. Then we generate two children for each parent pair that is a combination
#of the two parent's mazes. We accomplish this by splicing the two parents using a random integer for the first
#child and another random integer for the second child. After we create mazes for the children, we introduce
#the idea of mutation using random integers (a,b) that block off cell(a,b) in the child maze if its not already blocked.
#After generating the children, we sort the list of the population's mazes by the 'hardness' metric attribute which can be 
#changed based on the metric we're using. We then only keep the n best mazes in the population and continue the process.
#It's important to shuffle the list of population mazes at the start of every generation because if we only pair hard to 
#solve mazes with each other, their children will likely be unsolvable and need to be discarded. 
def generatehardmazes(dim,p):
    mazes=[]
    for i in range(200):
        block=createpointer(generatemaze(dim,p))
        pathsize=DFS(block)
        
        if pathsize != 'Failure':
            mazereset(block)
            maze=Maze(block,pathsize)
            mazes.append(maze)
    mazes.sort(key=lambda x: x.path,reverse=True)
    generation=1
    generationlst=[1]
    pathlen=[]
    initiallen=[]
    for i in mazes:
        initiallen.append(i.path)
    pathlen.append(sum(initiallen)/len(initiallen))
    
        
    while generation<100:
        if len(mazes)%2 != 0:
            mazes.pop()
        random.shuffle(mazes)
        numparents=len(mazes)
        for i in range(0,len(mazes)-1,2):
            parent1=mazes[i].maze
            parent2=mazes[i+1].maze
           
            
            rand1=random.randint(1,dim-1)
            rand2=random.randint(1,dim-1)
            child1=parent1[:rand1]+parent2[rand1:]
            child2=parent1[:rand2]+parent2[rand2:]
            
            child1maze=[]
            child2maze=[]
            
            for i in child1:
                row=[]
                for j in i:
                    if j.isEmpty==False:
                        row.append(Cell(0))
                    else:
                        row.append(Cell(1))
                child1maze.append(row)
            for i in child2:
                row=[]
                for j in i:
                    if j.isEmpty==False:
                        row.append(Cell(0))
                    else:
                        row.append(Cell(1))
                child2maze.append(row)
            
            
            
            mutation1=(random.randint(0,dim-1),random.randint(0,dim-1))
            mutation2=(random.randint(0,dim-1),random.randint(0,dim-1))
            
            while mutation1==(0,0) or mutation1==(dim-1,dim-1):
                mutation1=(random.randint(0,dim-1),random.randint(0,dim-1))
            while mutation2==(0,0) or mutation2==(dim-1,dim-1):
                mutation2=(random.randint(0,dim-1),random.randint(0,dim-1))
            
    
            
            #These mutations are "random" in nature. It's possibe that reversing the 
            #blocked status of a cell can lead to a negligible, negative, or positive 
            #effect on the hardness of the maze. Doing so can open a path to an easier
            #path to the goal, not affect the path to the goal, or force the algorithm
            #to take a longer path to the goal depending the specific maze case
        
            if child1maze[mutation1[0]][mutation1[1]].isEmpty==False:
                child1maze[mutation1[0]][mutation1[1]].isEmpty=True
            else:
                child1maze[mutation1[0]][mutation1[1]].isEmpty=False
            
            if child2maze[mutation2[0]][mutation2[1]].isEmpty==False:
                child2maze[mutation2[0]][mutation2[1]].isEmpty=True
            else:
                child2maze[mutation2[0]][mutation2[1]].isEmpty=False
            
            
            
            
            
            
            child1maze=createpointer(child1maze)
            child2maze=createpointer(child2maze)
            child1path=DFS(child1maze)
            child2path=DFS(child2maze)
    
            if child1path != 'Failure':
                mazereset(child1maze)
                mazes.append(Maze(child1maze,child1path))
            if child2path != 'Failure':
                mazereset(child2maze)
                mazes.append(Maze(child2maze,child2path))
                
        
        
        mazes.sort(key=lambda x: x.path,reverse=True)
        mazes=mazes[:numparents]
        generationlst.append(generation+1)
        pathlens=[]
        for i in mazes:
            pathlens.append(i.path)
        pathlen.append(sum(pathlens)/len(pathlens))
        generation += 1
    print(mazes[0].path)
    printmaze(mazes[0].maze)
    hardplotter(generationlst,pathlen)
def hardplotter(a,b):
    plt.plot(a,b)
    plt.xlabel('generations')
    plt.ylabel('average max fringe size of population')
    plt.show()
def printmaze(maze):
    mazelst=[]
    for i in maze:
        row=[]
        for j in i:
            if j.isEmpty==False:
                row.append('X')
            else:
                row.append('O')
        mazelst.append(row)
    for i in mazelst:
        print(*i)
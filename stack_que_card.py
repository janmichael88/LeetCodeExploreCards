#################################
#Design Circular Queue 08/19/20
#################################
class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.q = [0]*k
        self.head = 0
        self.tail = 0
        

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """ 
        if self.isFull():
            return False
        
        self.q[self.tail % len(self.q)] = value
        self.tail += 1
        return True
        

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.isEmpty():
            return False
        self.head += 1
        return True

        

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.q[self.head % len(self.q)] #there could be overflow with the pointers
            
        
        

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.q[(self.tail-1) % len(self.q)]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        #when both pointers are pointing to the same thing
        return self.head == self.tail
        
        

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        #its full when the tail has caught up to the head
        return self.tail - self.head == len(self.q)
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()

#####################################
#346. Moving Average from Data Stream
#####################################
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = []
        
    def next(self, val: int) -> float:
        size, queue = self.size, self.queue
        queue.append(val)
        # calculate the sum of the moving window
        window_sum = sum(queue[-size:])

        return window_sum / min(len(queue), size)

from collections import deque
class MovingAverage(object):
    '''
    we could use a double ended q or a deque
    when we append we pop the old element we only start popping one the size of the q gets to tis ze, 
    the space now is O(size) instead of O(M), which would grow at each invocation of next
    when doing sum update keep sum from previous window and add in the new element
    this reduces the time compelxity to constant operation O(val)
    '''

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.size = size
        self.q = deque()
        
        #keeping tracking of sum and count so far
        self.window_sum = 0
        self.count = 0
        

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.count += 1
        self.q.append(val)
        #get the old element that causes the q to be larger than the sieze
        tail = self.q.popleft() if self.count > self.size else 0
        
        #get the new sum
        self.window_sum = self.window_sum - tail + val
        
        return float(self.window_sum) / float(min(self.size,self.count))


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)

class MovingAverage(object):
    '''
    using a circular quere, where it elimnates the need to have two pointers
    the oldest element is automtically remove - we had to explicity define this with the deque
    tail = (head+1) mod size
    '''

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.size = size
        self.q = [0]*self.size
        self.head = self.window_sum = 0
        self.count = 0
        

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.count += 1
        #calculate the new sum by shifting the window
        tail = (self.head + 1) % self.size
        self.window_sum = self.window_sum - self.q[tail] + val
        
        #now move the head
        self.head = (self.head + 1) % self.size
        self.q[self.head] = val
        
        return float(self.window_sum) / min(self.size, self.count) #remember this logic instead of doing an if else



###########################
#Walls and Gates
###########################
class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        """
        '''
        what if i scan the 2d array first for all gates, storing their i,j locations
        i should to the same for walls
        then go left and right from each of the gates and turning INF to the distance from the gate
        what if i tried to reach each INF from each gate, do this for all gates and just re input the min?
        thats dumb
        strategy, go through the 2d adding the i,j locatinos of all the gates
        change the i,j of the elements that can be reached in the firs round
        add those back to the q
        increment for the next round, this will make more sense when i code it tout
        '''
        inf = 2147483647
        gateLocations = []
        
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        
        for i in range(0,len(rooms)):
            for j in range(0,len(rooms[0])):
                if rooms[i][j] == 0:
                    gateLocations.append([i,j])
        #start off with INFS one away, increment when done
        count = 1
        
        while True:
            gateMovements = 0 #counting the number of swaps from INF to count
            newGateLocations = []
            
            for gate in gateLocations:
                for x,y in directions:
                    dx = gate[0] + x
                    dy = gate[1] + y
                    
                    #check if in bounds
                    if (dx>-1 and dx<len(rooms)) and (dy>-1 and dy<len(rooms[0])):
                        if rooms[dx][dy] == inf:
                            #change
                            rooms[dx][dy] = count
                            #add to new locations
                            newGateLocations.append([dx,dy])
                            gateMovements = 1 #since we could change a gate
            gateLocations = newGateLocations
            count += 1
            
            if gateMovements ==0:
                break

from collections import deque
class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        """
        '''
        what if i scan the 2d array first for all gates, storing their i,j locations
        i should to the same for walls
        then go left and right from each of the gates and turning INF to the distance from the gate
        what if i tried to reach each INF from each gate, do this for all gates and just re input the min?
        thats dumb
        strategy, go through the 2d adding the i,j locatinos of all the gates
        change the i,j of the elements that can be reached in the firs round
        add those back to the q
        increment for the next round, this will make more sense when i code it tout
        '''
        if not rooms:
            return
        inf = 2147483647
        rows = len(rooms)
        cols = len(rooms[0])
        
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        q = deque()
        
        for i in range(0,rows):
            for j in range(0,cols):
                if rooms[i][j] == 0:
                    q.append([i,j,0])                         
        
        while q:
            x,y,s = q.popleft()

            for dx,dy in directions:
                nx = x + dx
                ny = y + dy

                #check if in bounds
                if (0<= nx < rows) and (0<=ny<cols) and(rooms[nx][ny]==inf):
                    rooms[nx][ny] = s+1
                    #add to new locations
                    q.append([nx,ny,s+1])

#######################
#Number of Islands 08/28/2020
#######################
from collections import deque
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        '''
        use DFS
        treat each element as a node with at most two edges
        scan the 2d array and is there a 1, trigger DFS
        every visited node should be set as "O" to makr as a visited node
        count the number of root nodes that rigger DFS, which would be the number of islans
        the idea here is to trigger a DFS call and up the count when it fires, and when it fires dfs to make the island by adding it to a visited set
        '''
        if not grid or not grid[0]:
            return 0
        
        num_islands = 0
        visited = set()
        rows,cols = len(grid),len(grid[0])
        directions  = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        
        def dfs(r,c):
            if (r not in range(rows)) or (c not in range(cols)) or (grid[r][c] == '0') or ((r,c) in visited):
                return
            visited.add((r,c))
            #recurse
            for dx,dy in directions:
                dfs(r+dx,c+dy)
        
        #invoke
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "1" and (i,j) not in visited:
                    num_islands += 1
                    dfs(i,j)
        return num_islands


#######################
# Open The Lock
########################
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        '''
        from the hint, this is a graph traversal problem, in fact find the shortest path problem
        there are 100000 different states i.e nodes,
        each node conected by an edge which is the turn of a dial
        i can't go into the state if its in dead ends
        dfs or bfs, BFS obvs! would find the shortes path...dfs would enumerate all possoble paths
        
        to solve the shortest path problem we need to use bfs
        useing a q and a seen set
        
        we define a neighbors functions, for each position in the lock (0,1,2,3) for each of the turns d (-1,1) we determine the vlaues of the lock after the ith wheel has been turned in the d direction
        
        edge cases:
            make sure we do no traverse and edge that leads to dead end, and we must also add, '0000' in the beginning
        '''
        def neighbors(node):
            #get nodes differing by one turn on the dial
            for i in range(4):
                x = int(node[i])
                for d in (-1,1):
                    y = (x + d) % 10
                    yield node[:i] + str(y) + node[i+1:]
                    
        dead = set(deadends)
        seen = {'0000'}
        q = deque([('0000',0)])
        
        while q:
            node,depth = q.popleft()
            if node == target:
                return depth
            if node in dead:
                continue
            for n in neighbors(node):
                if n not in seen:
                    seen.add(n)
                    q.append((n,depth+1))
                    
        return -1


#######################
# Perfect Squares
#######################

#long article, go through each one finely
#recursion
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        this is a long article, so lets go over each of the solutions
        recursion!
        this is a combindation problem so we could use some kind of backtracking given our list of perfect squares up to n, exmplae generate all permutations in range(len(square_nums))
        if any of those permutations add to n its a valid comb
        num_squares(n) = min(num_squares[n-1] + 1)
        '''
        square_nums = set([i**2 for i in range(1,int(math.sqrt(n))+1)])
        
        def dfs(num):
            if num in square_nums:
                return 1
            min_elements = float('inf')
            
            #recurse
            for square in square_nums:
                if num < square:
                    #can't use it
                    continue
                #take
                new_num = dfs(num-square) + 1
                min_elements = min(min_elements,new_num)
            return min_elements
        
        return dfs(n)

# dp solution
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        dp solution
        this is a long article, so lets go over each of the solutions
        recursion!
        this is a combindation problem so we could use some kind of backtracking given our list of perfect squares up to n, exmplae generate all permutations in range(len(square_nums))
        if any of those permutations add to n its a valid comb
        num_squares(n) = min(num_squares[n-1] + 1)
        '''
        square_nums = set([i**2 for i in range(1,int(math.sqrt(n))+1)])
        
        dp = [float('inf')]*(n+1)
        
        dp[0] = 0
        
        for i in range(n+1):
            for sq in square_nums:
                if i < sq:
                    continue
                dp[i] = min(dp[i],dp[i-sq] + 1)
        return dp[-1]

class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we could use a greedy approach with recursion
        starting from the combination of one single number to the multiple numbers,
        once we find a combinations that can sum up to n, we can say we must have formed the smallest combindation, since we enumerate the combinations greedily from small to large
        define func is_divided_by(n,count)
        thie returns a boolean to indicate whether the number n can be divied by a combination with count of square numbers, rather than returning the exact size 
        num_saure = argmin_{count} (is_divided_by(n,count))
        for sample n = 5, candidiates = [1,4]
        func(5,1) nope!
        funct(5,2) fire
        funct(5-1,1) func(5-4,1)
        
        why? proof by contraction
        suppose we find a count m that divied n
        then we find a candidate p, after m that also divides n
        p < m, meaning p would have beend found before m
        '''
        square_nums = set([i**2 for i in range(1,int(math.sqrt(n))+1)])
        def dfs(n,count):
            #returns true of n can be decomposed into 'count' num perf squares
            #ex dfs(12,3) true
            #ex dfs(12,2) false
            
            if count == 1:
                return n in square_nums
            
            for k in square_nums:
                if dfs(n-k,count-1):
                    return True
            return False
        
        for i in range(1,n+1):
            if dfs(n,i):
                return i

class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        recall the greedy enumeration creats an N-ary tree, which means we could use BFS
        given an N-ary tree, each node represents a fuction call:
        representing the remainder when n is divided by a square
        out task woudl to find the node in the tree, where
        the val of the node is a square number and the distance between the root and the node is minimal
        algo
            generate list of perfect squares up to N
            q up, keeping in reaminders
            at each iteration, check if the remainder is also a square
            if ther emainder is not, subtract it with one of the square numbers to update the remainder, 
            put the new remaninder back into q
            we break out of the loop once we encouter another square
        '''
        #generate squares
        squares = []
        sq = 1
        while sq*sq <= n:
            squares.append(sq*sq)
            sq += 1
        
        level = 0
        q = set([n])
        while q:
            level += 1
            next_q = set()
            for rem in q:
                for sq in squares:
                    if rem in squares:
                        return level
                    elif rem < sq:
                        continue
                    else:
                        next_q.add(rem-sq)
            q = next_q
        
        return level

#############################
#  Valid Parentheses
#############################
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        load each char onto a stack only if its opening
        check of top is closing bracket and pop it off
        if its not closing, return False
        if we encounter an opening bracket we push onto the stack
        if we later encounter a closingbrackert then we check the top elemnt  is an opneindg bracket of the char we are on, if it is we pop it off, else its an invalid exprssion
        at the end were are left with items in stack which mean it is in invalid expression
        '''
        stack = []
        mapping = mapping = {")": "(", "}": "{", "]": "["}
        
        for ch in s:
            #check if in mapping
            if ch in mapping:
                #view the twop
                top = stack.pop() if stack else "#" #only pop if there is a stack
                #check closting
                if mapping[ch] != top:
                    return False
            else:
                stack.append(ch)
                    
        return not stack

#############################
# Daily Temperatures
###############################
#TLE
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        '''
        the naive way would be to just start at each one with another pointer advancing
        to find a higher temp, start with that
        '''
        N = len(T)
        results = [0]*N
        for i in range(N):
            j = i + 1
            while j < N:
                if T[j] > T[i]:
                    break
                else:
                    j += 1
            if j == N:
                results[i] = 0
            else:
                results[i] = j - i
        return results


#from the card, we need to use a stack
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        '''
        [1,1,4,]
        [73,74,75,71,69,72]
        
        keep pushing on to a stack until the element we add is greater than what we are point at
        if it 
        we can use the next array
        since the temps are in a range of [30,100]
        just find the next day at which we have a high temp
        we can use a stack
        we load onta the stack [idx,temp]
        and whenever the op of the stack is less then the value
        we need to keep track of how many have passed since there was a new hihger value
        
        '''
        results = [0]*len(T)
        stack = []
        
        for i,v in enumerate(T):
            #check wheter current value is greater than the last appended stack value
            #pop all elemnts smaller than current
            while stack and stack[-1][1] < v:
                idx, temp = stack.pop()
                #check how many indcies have passed since we last have a samller temp
                #we compare all stack element befroe inserting
                results[idx] = i - idx
            stack.append([i,v])
        return results
        

####################################
#Evaluate Reverse Polish Notation
###################################
#well it got half, not bad, but look at the solution
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        '''
        well reverse polish notation if just pst fix notation
        https://www.youtube.com/watch?v=qN8LPIcY6K4&ab_channel=BackToBackSWE
        good theory video
        just use a stack
        pop off two items evaluate and push back
        return the remaining element in the stack
        '''
        def calc(num1,num2,op):
            if op == "+":
                return num1+num2
            if op == "-":
                return num1-num2
            if op == "*":
                return num1*num2
            if op == "/":
                return num1//num2
        stack = []
        N = len(tokens)
        ops = set(["+", "-", "*", "/"])
        
        for i in range(N):
            if tokens[i] in ops:
                num2 = int(stack.pop())
                num1 = int(stack.pop())
                stack.append(calc(num1,num2,tokens[i]))
            else:
                stack.append(tokens[i])
        
        return stack.pop()

####################
#Number of Islands
####################
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        '''
        i can do dfs call from each node
        if i am at a 1 see if i can reach other ones in all directions, and keep dfsing
        once i can't the call should end, and i made an island
        '''
        if not grid or not grid[0]:
            return 0
        
        islands = 0
        visited = set()
        rows = len(grid)
        cols = len(grid[0])
        
        def dfs(r,c):
            if r<0 or c<0 or r>=rows or c>=cols or (grid[r][c] == '0') or ((r,c) in visited):
                return
            visited.add((r,c))
            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)
            
        #invoke at each point
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "1" and (i,j) not in visited:
                    islands += 1
                    dfs(i,j)
                    
        return islands

################
#Clone Graph
################
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        define a cloned hash map outside the caler
        in our dfs function we visita node, give reference to a new node with the nodes val
        then for each of its neighbors add to the hash map if not visited
        '''
        if not node:
            return None
        
        visited = {}
        def dfs(node):
            #the only wrench we need to take care of is a cycle
            #to prevent a cycle we make a new node
            new = Node(node.val)
            #put into hasho
            visited[node.val] = new
            #assigne neighbors object
            new.neighbors = []
            
            for n in node.neighbors:
                if n.val not in visited:
                    new.neighbors.append(dfs(n))
                else:
                    new.neighbors.append(visited[n.val])
            return new
        
        return dfs(node)

#iteratively
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        iterativle using a stack
        i can use a hashamp for the clones
        where a key is the node and value of the copied node
        '''
        if not node:
            return node
        cloned = {node:Node(node.val)} #return cloned[node]
        stack = [node]
        while stack:
            curr = stack.pop()
            for neigh in curr.neighbors:
                if neigh not in cloned:
                    #add to our stack but also add its neighbors in our hash
                    stack.append(neigh)
                    cloned[neigh] = Node(neigh.val)
                #if we have seen it, add to its cloned node
                cloned[curr].neighbors.append(cloned[neigh])
        return cloned[node]

#recursively
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        iterativle using a stack
        i can use a hashamp for the clones
        where a key is the node and value of the copied node
        '''
        if not node:
            return node
        cloned = {node:Node(node.val)} #return cloned[node]
        
        def dfs(node):
            for n in node.neighbors:
                if n not in cloned:
                    cloned[n] = Node(n.val)
                    dfs(n)
                #if we have, index back into hash at the node and add its neighbors
                cloned[node].neighbors.append(cloned[n])
        dfs(node)
        return cloned[node]

#BFS

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return
        cloned = {node: Node(node.val)}
        q = deque([node])
        while q:
            current = q.popleft()
            for neigh in current.neighbors:
                if neigh not in cloned:
                    cloned[neigh] = Node(neigh.val)
                    q.append(neigh)
                cloned[current].neighbors.append(cloned[neigh])
            
        return cloned[node]


####################
# Target Sum
####################
#TLE
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        self.ways = 0
        N = len(nums)
        
        def rec_build(nums, index, summ,S):
            if index == len(nums):
                if summ == S:
                    self.ways += 1
            else:
                rec_build(nums,index+1,summ +nums[index],S)
                rec_build(nums,index+1,summ -nums[index],S)
        rec_build(nums,0,0,S)
        return self.ways

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        '''
        recursive with memoization
        '''
        self.ways = 0
        cache = {}
        def rec_build(i,currsum):
            #add in index and current sum to cache
            if (i,currsum) not in cache:
                #update our cache when we have use all elements
                if i == len(nums):
                    #a solution of elements with valid sum
                    if currsum == S:
                        cache[(i,currsum)] = 1
                    #not valid sum
                    else:
                        cache[(i,currsum)] = 0
                #if we we have not used all the elements keep recursing and adding to cache
                else:
                    cache[(i,currsum)] = rec_build(i+1,currsum+nums[i]) + rec_build(i+1,currsum-nums[i])
            return cache[(i,currsum)]

            
        return rec_build(0,0)

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        '''
        recursive with memoization
        another way
        https://www.youtube.com/watch?v=jF62VYElDHY&ab_channel=HappyCoding
        '''
        cache = {}
        N = len(nums)
        def rec_build(i,currsum):
            if (i,currsum) in cache:
                return cache[(i,currsum)]
            if i == N:
                if currsum == S:
                    return 1
                else:
                    return 0
            
            result = rec_build(i+1,currsum-nums[i]) + rec_build(i+1,currsum+nums[i])
            cache[(i,currsum)] = result
            return result
        
        return rec_build(0,0)

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        '''
        DP soltion
        https://www.youtube.com/watch?v=jF62VYElDHY&ab_channel=HappyCoding
        imagine we have the array [1,2,3,4,5] and S = 3
        we can have 1-2+3-4+5 = 3
        we can write
        1+3+5 = 3 + 4 + 5
        which is just sum(nums) + S // 2
        now its knapsack problem because we have the capacity and the items are the nmbers
        so how many ways can we get that capacity, 0 1 knapsakc problem translates to DP
        can do sanity check, ie if we cannot get a sum divisble by two, return 0
        1D dp array is too much, just start with the 2D dp array solution for now
        https://leetcode.com/problems/target-sum/discuss/804311/Python-DP-with-Comments
        
        '''
        if not nums:
            return 0
        s = sum(nums)
        if S > s:
            return 0 #impossible for a solution
        
        #let dp[i][j] represent the number of possible wayts to get sum j using the frist i numbers
        #note there are (s*2+1) possible sums, hence the number of columns
        dp = [[0 for _ in range(s*2+1)] for _ in range(len(nums))]
        
        #base case: using only the first number
        #this number locates at index s+nums[0] because the first s entries correspond to negatives numbers
        #so at that entry taking the first num i can at least get that
        dp[0][s+nums[0]] += 1
        dp[0][s-nums[0]] += 1
        
        #the general case
        #using the first i number
        for i in range(1,len(nums)):
            #for each sum [-s,s]
            for j in range(s*2+1):
                #when j - nums[i] >= 0 we can use and making we have valid indices
                if j - nums[i] >=0 and dp[i-1][j-nums[i]] > 0:
                    #use nums[i]
                    # We force ourselves to use num[i]. Then the complement is (j - nums[i]).
                    # There are dp[i-1][j-nums[i]] ways to get a sum of (j-nums[i]) using the first (i-1) numbers.  
                    # Hence, we increment the count dp[i][j] by this amount.
                    dp[i][j] += dp[i-1][j-nums[i]]
                #now for the compleement
                if j + nums[i] <= s*2 and dp[i-1][j+nums[i]] > 0:
                    # We force ourselves to use -num[i] (NOTE THE MINUS SIGN HERE!). Then the complement is (j + nums[i]).
                    # There are dp[i-1][j+nums[i]] ways to get a sum of (j+nums[i]) using the first (i-1) numbers.  
                    # Hence, we increment the count dp[i][j] by this amount.
                    dp[i][j] += dp[i-1][j+nums[i]]
                
        #our target locates teh index at s+S becasue the first s entires are negative
        return dp[-1][s+S]


#################################
#Binary Tree Inorder Traversal
###################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        recursive is too easy, but try it iteratively
        we keep a stack and while the stack isnt emptu and current node isn't nul
        we push the left on to the stack and go left
        we process the top node on the stack and set it right
        '''
        stack = []
        results = []
        current = root
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            #once gone all the way left look at stack
            current = stack.pop()
            results.append(current.val)
            current = current.right
        
        return results

###############################
# Implement Queue using Stacks
###############################
#actually go thourhg O(N) pop, but look over O(1) pop
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        '''
        we can use an auaxially stack
        any time we call push pop all elements from first stack back on
        then push x, then push back
        '''
        self.main = []
        self.aux = []
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        #clear out main 
        while self.main:
            self.aux.append(self.main.pop())
        #push x
        self.main.append(x)
        #push back
        while self.aux:
            self.main.append(self.aux.pop())
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.main.pop()
        

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.main[-1]
        

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        #see if it exists
        return not self.main
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


#################################
#Implement Stack Using Queues
##############################
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = deque()

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self.q.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.q.pop()
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.q[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return not self.q

class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #streamlingin to push as 0(1) and pop O(N)
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        #this i just O(1)
        self.q1.append(x)
        #give reference to top, last recently added
        self.top = x
    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        #clear q1 and dump elements to q2
        #result is last recenly popped
        #swap q1 and q2
        while len(self.q1) > 1:
            self.top = self.q1.popleft()
            self.q2.append(self.top)
        #get last element fromr q1
        result = self.q1.popleft()
        #swap
        self.q1, self.q2 = self.q2, self.q1
        return result
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.top

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return len(self.q1) == 0

###############
# Decode String
###############
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        we use a stack and keep pushing each char onto the stack
        we evaluate the stack when we have a closing bracket and decode
        we push the deocded string back in reverse order
        then pop off the stack into a string and reverse
        '''
        stack = []
        for i in range(len(s)):
            if s[i] == ']': #evaluate
                decoded = ''
                #keep popping intil openidng
                while stack[-1] != '[':
                    decoded += stack.pop()
                #remove last opening
                stack.pop()
                #now take care of coef
                base = 1 
                k = 0
                while stack and ord('0') <= ord(stack[-1]) <= ord('9'):
                    k = k + int(stack.pop())*base
                    base *= 10
                decoded *= k
                #now push back in reverse
                for j in range(len(decoded)-1,-1,-1):
                    stack.append(decoded[j])
            #else add in the char
            else:
                stack.append(s[i])
        
        #final output
        output = ''
        while stack:
            output += stack.pop()
        return output[::-1]

##############
#Flood Fill
##############
#DFS
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        seen = set()
        rows = len(image)
        cols = len(image[0])
        
        start_color = image[sr][sc]
        
        #if starting is already new color, i cant flood fill anything
        if start_color == newColor:
            return image
            
        def dfs(r,c):
            if r < 0 or r > rows-1 or c < 0 or c > cols-1:
                return
            if (r,c) not in seen:
                if image[r][c] == start_color:
                    image[r][c] = newColor
                    dfs(r,c+1)
                    dfs(r,c-1)
                    dfs(r+1,c)
                    dfs(r-1,c)
                    seen.add((r,c))
            else:
                return
            
        dfs(sr,sc)
        return image
#BFS
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        rows = len(image)
        cols = len(image[0])
        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        
        start_color = image[sr][sc]
        
        if start_color == newColor:
            return image
        
        q = deque([(sr,sc)])
        while q:
            curr_r,curr_c = q.popleft()
            for dirr in directions:
                #examine neighbors
                r = curr_r + dirr[0]
                c = curr_c + dirr[1]
                #check in bounds
                if 0 <= r < rows and 0<= c < cols and image[r][c] == start_color:
                    image[r][c] = newColor
                    #add in neighbors
                    q.append((r,c))
        return image 


##############
#01 Matrix
##############
class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        dfs for each i,j element, checking in all four directions first,
        if of these are zero the result if the distance
        every time i call dfs increment distance by 1, then store that distance at the i,j
        well dfs was stupud, start with bfs instead
        we only want to find the distance if we hit a one
        we can pass over the matrix and perform bfs when we hit a one
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        def bfs(node):
            #node is a tuple
            i,j = node
            q = deque([(i,j,0)]) #eleemnts is (x,y,dist), of course starting from zero
            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            visited = set()
            
            while q:
                x,y,d = q.popleft()
                #we keep going until we hit a zero
                if matrix[x][y] == 0:
                    return d
                #add to visited 
                visited.add((x,y))
                #otherwise bfs
                for dirr in directions:
                    newx,newy = x+dirr[0],y+dirr[1]
                    #boundary check
                    if 0 <= newx < rows and 0 <= newy < cols:
                        #not visited gain
                        if (newx,newy) not in visited:
                            q.append((newx,newy,d+1))
            return -1
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 1:
                    d = bfs((i,j))
                    matrix[i][j] = d
        return matrix

class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        instead of invoking bfs from a cell who's entry is 1, we can flip the problem
        instead of thinking 1 and reaching to zero, thing of starting at all the zeros and reach the one
        this allows a single BFS seaach that emerges from different places (all the zeros)
        algo:
            traverse the amtrix and dump into q the i,j places that have a zero
            bfs on these elements in the q
            if a enigh cell s not been visited: then it must be a 1 cell (since we passed zeros first)
            append the neighor cell in the q and mutate the grid
        '''
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        q = deque()
        visited = set()
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    visited.add((i,j))
                    q.append((i,j))
                    
        #remeber, not the nodes not in the set should all be 1
        #we inestiage each nodes neighbors and if they aren't in visited, they must be 1, so we increment that i,j locaiton by 1
        while q:
            x,y = q.popleft()
            for dirr in directions:
                newx,newy = x+dirr[0], y+dirr[1]
                if 0 <= newx < rows and 0 <= newy < cols and (newx,newy) not in visited:
                    matrix[newx][newy] = matrix[x][y] + 1 #one away from the current x and y
                    visited.add((newx,newy))
                    q.append((newx,newy))
        
        return matrix

#################
#Keys and Rooms
##################
#easy peezy :)
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        '''
        bfs starting from index 0
        once im done return of the lenth of my visited set is the same is the number of rooms
        '''
        visited = set()
        N = len(rooms)
        q = deque([0])
        
        while q:
            current = q.popleft()
            for key in rooms[current]:
                if key not in visited:
                    q.append(key)
            visited.add(current)
        
        return len(visited) == N
        
###############################
#RPrint everse A String Recursively
##############################
def print_reverse(s,idx):
    if idx >= len(s):
    	return
    print_reverse(s,idx+1)
    #note this would print line be line
    print(s[idx])

print_reverse('abcde',0)


###############
#Reverse a string
#################
#two pointer method first
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        l,r = 0, len(s) - 1
        while l <= r:
            s[l],s[r] = s[r],s[l]
            l += 1
            r -= 1
        
#revursively
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        def rec_reverse(s,start,end):
            if start >= end:
                return
            s[start],s[end] = s[end],s[start]
            rec_reverse(s,start+1,end-1)
            
        rec_reverse(s,0,len(s)-1)

##############
#All Subsets
#############
#this was just an aside to get more practice with recursion

test = [1,2,3,4]

def rec_create(array,idx,path):

    if idx >= len(array):
        print(path)
        return
    path.append(array[idx])
    rec_create(array,idx+1,path)
    #backtrack
    path.pop()
    rec_create(array,idx+1,path)

rec_create(test,0,[])


#now adding to a list and returning all subsets
test = [1,2,3,4]
def back_tracking(array):

    output = []
    def rec_create(array,idx,path):
        if idx >= len(array):
            output.append(path[:])
            return
        path.append(array[idx])
        rec_create(array,idx+1,path)
        #backtrack
        path.pop()
        rec_create(array,idx+1,path)

    rec_create(array,0,[])
    return(output)

print(back_tracking(test))


#now cascading
test = [1,2,3,4]

output = [[]]

for num in test:
    output += [[num] + foo for foo in output]

print(output)


#######################
#Swap Nodes in Paires
########################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        before doing this recursively, lets try iteratively
        '''
        if not head or not head.next:
            return head
        dummy = ListNode()
        dummy.next = head
        curr = dummy
        #loop invariant we need to keep checking the next and .next.nest
        while curr.next and curr.next.next:
            #give reference to first and second
            first = curr.next
            second = curr.next.next
            #swap
            first.next = second.next
            second.next = first
            curr.next = second
            #advance two
            curr = curr.next.next
        
        return dummy.next

#recursively
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        recusively
        '''
        def swap(head):
            if not head or not head.next:
                return head
            first = head
            second = head.next
            
            first.next = swap(second.next)
            second.next = first
            return second
        return swap(head)


#######################
#Reverse Linked List
######################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        while traversing just set the next pointers node to previous
        since the a node does not have a reference to previous node, we need to store it
        you also need a pointer to store the next node before changing the reference
        '''
        prev = None
        cur = head
        while cur:
            #store next
            nextt = cur.next
            #connect current next to prvious
            cur.next = prev
            #move prev up
            prev = cur
            #move up cur
            cur = nextt
        return prev # we need to return prev because it holds all the previous currs



#Recursively
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        assum we have m nodes and we are at k
        n_1 -> ... -> n_k -> n_{k+1} ... -> n_{m} -> None
        assume we are at node k, and everything after k has been reversee
        we then would wan n_k.next.next point to n_k
        '''
        def rec_reverse(node):
            #base case
            if not node or not node.next:
                return node
            
            #reverse the rest of the list
            rev = rec_reverse(node.next)
            node.next.next = node
            #make sure to poin the final node to next
            node.next = None
            return rev
        
        return rec_reverse(head)


##################################
# Search in a Binary Serach Tree
##################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        '''
        iteratively
        '''
        q = deque([root])
        while q:
            node = q.popleft()
            if node:
                if node.val == val:
                    return node
                if node.val > val:
                    q.append(node.left)
                else:
                    q.append(node.right)
                    
        return None

#recursively
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        '''
        recursively now
        
        '''
        def dfs(node):
            if not node or node.val == val:
                return node
            if node.val < val:
                return dfs(node.right)
            else:
                return dfs(node.left)
        return dfs(root)

#######################
#Pascals' Triangle II
#######################
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        lets do this a couple of differnt ways before doing recursively
        iterativley we only need the previous to get the next row
        '''
        if rowIndex == 0:
            return [1]
        
        previous = [1]
        nextt = None
        for i in range(1,rowIndex+1):
            nextt = [1]*(i+1)
            for j in range(1,len(nextt)-1):
                nextt[j] = previous[j-1] + previous[j]
            
            previous = nextt
        return nextt

#MATH SOLUTION
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        we can just use the bionmial coeffiecnts at a rows
        at the nth row we just have i C n, where i i the index and n is the row number
        '''
        import math
        def combinations(n,k):
            return math.factorial(n) / (math.factorial(k)*math.factorial(n-k))
        
        return [combinations(rowIndex,i) for i in range(rowIndex+1)]

#recursively
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        recursively....ohh boyyyy
        we can define a function, call rec
        we know that the value at a row and col is the sum of the two elements abover that row to th left and above that row to the dright
        so rec(r,c) = rec(r-1,c-1) + rec(r-1,c)
        the base case is when rec(r,c) == 1 when r == 1 or r == c, the last item for the ith row
        '''
        #make starting row at 1
        rowIndex += 1
        memo = {}
        def rec_pascal(row,col):
            #beginning of or end of row
            if col == 1 or row == col:
                return 1
            return rec_pascal(row-1,col-1) + rec_pascal(row-1,col)
        
        result = []
        for i in range(1,rowIndex+1):
            result.append(rec_pascal(rowIndex,i))
        
        return result

#recursively with memo

class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        recursively....ohh boyyyy
        we can define a function, call rec
        we know that the value at a row and col is the sum of the two elements abover that row to th left and above that row to the dright
        so rec(r,c) = rec(r-1,c-1) + rec(r-1,c)
        the base case is when rec(r,c) == 1 when r == 1 or r == c, the last item for the ith row
        '''
        #make starting row at 1
        rowIndex += 1
        mem = {}
        def rec_pascal(row,col):
            #beginning of or end of row
            if col == 1 or row == col:
                return 1
            if (row-1, col-1) not in mem:
                mem[(row-1, col-1)] = rec_pascal(row-1, col-1)
            v1 = mem[(row-1, col-1)]

            if (row-1, col) not in mem:
                mem[(row-1, col)] = rec_pascal(row-1, col)
            v2 = mem[(row-1, col)]

            return v1 + v2
        
        result = []
        for i in range(1,rowIndex+1):
            result.append(rec_pascal(rowIndex,i))
        
        return result

###################################
#Duplicate Calculation in Recursion
###################################
'''
we define the Fib sequence recursivel
F(n) = F(n-1) + F(n-2)
the base cases cases are
F(0) = 0
F(1) = 1
'''
import time
def fib(n):
    if n < 2:
        return(n)
    return fib(n-1) + fib(n-2)
start = time.time()
print(fib(35))
print(time.time()-start)


#now with meemo
import time
memo = {}
def fib(n):
    if n < 2:
        return(n)
    if n in memo:
        return(memo[n])
    result = fib(n-1) + fib(n-2)
    memo[n] = result
    return(result)
start = time.time()
print(fib(35))
print(time.time()-start)


#####################
#Fibonnaci Number
####################
#top down memo
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        top down memo
        '''
        memo = {}
        def rec_fib(n):
            if n < 2:
                return n
            if n in memo:
                return memo[n]
            result = rec_fib(n-1) + rec_fib(n-2)
            memo[n] = result
            return result
        
        return rec_fib(n)

#bottom up DP

class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        bottom up dp O(1) space
        
        '''
        if n < 2:
            return n
        n_1 = 0
        n_2 = 1
        for i in range(n-1):
            nextt = n_1 + n_2
            n_1 = n_2
            n_2 = nextt
        return nextt

#################
#Climbing Stairs
################
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        this is just a recursion
        the number of ways to the nth step is
        F(n) = F(n-1) + F(n-2)
        '''
        memo = {}
        def rec(n):
            if n < 3:
                return n
            if n in memo:
                return memo[n]
            result = rec(n-1) + rec(n-2)
            memo[n] = result
            return result
        
        return rec(n)

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        dynamic programing O(n)
        '''
        if n == 1:
            return 1
        dp = [0]*(n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
        
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        dynamic programing O(1)
        '''
        if n < 3:
            return n
        
        first = 1
        second = 2
        for i in range(3,n+1):
            third = first + second
            first = second
            second = third
        return third

############################
#Time Complexity - Recursion
############################
'''
O(T) is the number of recursion inovcations times the time complectiy of the recursive call
O(T) = R*O(S)
'''

#############################
#Space Complexity - Recursion
#############################
'''
As suggested by the name, the non-recursion related space refers to the memory space that is not directly related to recursion, which typically includes the space (normally in heap) that is allocated for the global variables.

Recursion or not, you might need to store the input of the problem as global variables, before any subsequent function calls. And you might need to save the intermediate results from the recursive calls as well. The latter is also known as memoization as we saw in the previous chapters. For example, in the recursive algorithm with memoization to solve the Fibonacci number problem, we used a map to keep track of all intermediate Fibonacci numbers that occurred during the recursive calls. Therefore, in the space complexity analysis, we must take the space cost incurred by the memoization into consideration.  
'''

########################
# Tail Recursion
########################
'''
when the return of a subrouter is another call recursively
'''
#exmplae of non_tail+reucroin
def sum_non_tail_rec(array):
    if len(array) == 0:
        return 0
    #notice how i need to add another element to the recursive call
    #the return is NOT JUST A isngle call
    return array[0] + sum_non_tail_rec(array[1:])

print(sum_non_tail_rec(list(range(5))))

#example tail recursion
#this is usually the way i write my recursive calls
def sum_tail_rec(array):
    def helper(array,accum):
        if len(array) == 0:
            return(accum)
        return(helper(array[1:],accum+array[0]))

    return(helper(array,0))

print(sum_tail_rec(list(range(5)))) 












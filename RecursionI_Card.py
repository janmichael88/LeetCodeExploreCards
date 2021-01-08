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
        





















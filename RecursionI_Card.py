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
###################
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
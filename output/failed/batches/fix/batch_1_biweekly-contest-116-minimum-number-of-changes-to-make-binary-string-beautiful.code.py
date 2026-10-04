class Solution:
    def minChanges(self, s: str) -> int:
        i = 0
        j = 1
        cnt = 0
        while j < len(s):
            if s[i] == s[j]:
                i += 2
                j += 2
            else:
                cnt += 1
                i += 1
                j += 2
        return cnt

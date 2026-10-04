from typing import List


class Solution:
    def maxSpending(self, values: List[List[int]]) -> int:
        r = []
        i = 0
        while i < len(values):
            r.extend(values[i])
            i += 1
        r = sorted(r)
        ans = 0
        i = 1
        while i <= len(r):
            val = r[-i]
            ans += i * val
            i += 1
        return ans

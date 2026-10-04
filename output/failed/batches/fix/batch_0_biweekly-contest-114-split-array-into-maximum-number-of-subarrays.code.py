from typing import List


class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        cur = -1
        c = 0
        for x in nums:
            cur = cur & x
            if cur == 0:
                c += 1
                cur = 0
        if c == 0:
            return 1
        return c

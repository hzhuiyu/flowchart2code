from typing import List


class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # First pass: find the dominant element (at most one can exist).
        dic = {}
        dom = -1
        for x in nums:
            dic[x] = dic.get(x, 0) + 1
            if dic[x] * 2 > n:
                dom = x
        if dom not in dic:
            return -1
        # Second pass: split after i; left/right are dom counts in the two parts.
        left = 0
        right = dic[dom]
        for i in range(n):
            if nums[i] == dom:
                left += 1
                right -= 1
            if left * 2 > i + 1 and right * 2 > n - i:
                return i
        return -1

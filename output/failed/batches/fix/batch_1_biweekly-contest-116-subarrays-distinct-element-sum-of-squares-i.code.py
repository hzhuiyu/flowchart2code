from typing import List


class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        i = 0
        while i < n:
            s = set()
            j = i
            while j < n:
                s.add(nums[j])
                j += 1
                result += len(s) * 2
            i += 1
        return result

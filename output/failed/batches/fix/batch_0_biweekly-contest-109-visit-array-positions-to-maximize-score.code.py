from typing import List


class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        best_even = -float('inf')
        best_odd = -float('inf')
        if nums[0] % 2 == 0:
            best_even = nums[0]
        else:
            best_odd = nums[0]
        for i in range(1, len(nums)):
            v = nums[i]
            if v % 2 == 1:
                cur = best_even + v
                if best_odd + v - x > cur:
                    cur = best_odd + v - x
                if cur > best_even:
                    best_even = cur
            else:
                cur = best_odd + v
                if best_even + v - x > cur:
                    cur = best_even + v - x
                if cur > best_odd:
                    best_odd = cur
        return max(best_even, best_odd)

from typing import List


class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        p = sum(nums)
        found = False
        nums.sort(reverse=True)
        i = 0
        # Walk from the largest side down; a suffix of >= 3 sides forms a
        # polygon once the largest remaining side is smaller than the sum
        # of the sides after it.
        while i < len(nums) - 2:
            v = nums[i]
            if v <= p - v:
                found = True
                break
            p -= v
            i += 1
        return p if found else -1

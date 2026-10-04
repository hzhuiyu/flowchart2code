from typing import List


class Solution:
    def isGood(self, nums: List[int]) -> bool:
        nums = sorted(nums)
        n = len(nums) - 1
        ok = len(nums) >= 2
        i = 0
        while ok and i < len(nums) - 1:
            expected = i + 1 if i < n - 1 else n
            if nums[i] == expected:
                i += 1
            else:
                ok = False
        return ok

from typing import List


class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        nums = [0] + nums + [float('inf')]
        n = len(nums)
        i = 0
        while i < n - 2 and nums[i] < nums[i + 1]:
            i += 1
        if i >= n - 2:
            return (n - 2) * (n - 1) // 2
        j = n - 1
        while j - 1 >= 1 and nums[j - 1] < nums[j]:
            j -= 1
        l = 0
        r = j
        res = 0
        while l < i:
            while r < n and nums[l] >= nums[r]:
                r += 1
            res += n - r
            l += 1
        return res

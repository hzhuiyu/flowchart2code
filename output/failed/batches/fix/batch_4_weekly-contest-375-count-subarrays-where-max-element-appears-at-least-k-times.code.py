from typing import List


class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        maxVal = max(nums)
        count = 0
        left = 0
        res = 0
        for right in range(len(nums)):
            if nums[right] == maxVal:
                count += 1
                while count > k:
                    if nums[left] == maxVal:
                        count -= 1
                    left += 1
            res += left
        return res

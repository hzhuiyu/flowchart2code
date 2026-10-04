from typing import List


class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        seen = set()
        ans = len(nums)
        for i in range(len(nums) - 1, -1, -1):
            v = nums[i]
            if 1 <= v < k:
                seen.add(v)
                if len(seen) == k:
                    ans = len(nums) - i
                    break
        return ans

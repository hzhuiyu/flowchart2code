from typing import List


class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        for i in range(n):
            dp[i] = nums[i]
            b = nums[i] - i
            best = 0
            for j in range(i):
                if nums[j] - j < b and dp[j] > best:
                    best = dp[j]
            dp[i] += best
        return max(dp)

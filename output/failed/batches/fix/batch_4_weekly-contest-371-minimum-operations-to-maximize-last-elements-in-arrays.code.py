from typing import List


class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        inf = 10**9
        dp = [[inf, inf] for _ in range(n)]
        i = n - 2
        for k in range(n):
            dp[k][0] = 0
            dp[k][1] = 0
        while i >= 0 and i < n - 1:
            dp[i][0] = inf
            dp[i][1] = inf
            if nums1[i] <= nums1[n - 1] and nums2[i] <= nums2[n - 1]:
                dp[i][0] = min(dp[i][0], dp[i + 1][0])
                dp[i][1] = min(dp[i][1], dp[i + 1][1] + 1)
            if nums1[i] <= nums2[n - 1] and nums2[i] <= nums1[n - 1]:
                dp[i][0] = min(dp[i][0], dp[i + 1][0] + 1)
                dp[i][1] = min(dp[i][1], dp[i + 1][1])
            i -= 1
        best = min(dp[0][0], dp[0][1])
        if best > n:
            return -1
        return best

from typing import List


class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [0]
        for v in nums:
            pre.append(pre[-1] + v)
        dp = {0: (0, 0)}
        i = 1
        while i <= n:
            j = 0
            while j < i:
                last = pre[i] - pre[j]
                if last >= -dp[j][1]:
                    dp[i] = max(dp.get(i, (0, 0)), (dp[j][0] + 1, last))
                j += 1
            i += 1
        return dp[n][0]

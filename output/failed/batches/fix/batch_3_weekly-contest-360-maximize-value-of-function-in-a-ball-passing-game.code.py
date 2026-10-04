from typing import List


class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        up = [[0] * 40 for _ in range(n)]
        dp = [[0] * 40 for _ in range(n)]

        for i in range(n):
            up[i][0] = receiver[i]
            dp[i][0] = receiver[i]

        for j in range(1, 40):
            for i in range(n):
                j1 = up[i][j - 1]
                up[i][j] = up[j1][j - 1]
                dp[i][j] = dp[i][j - 1] + dp[j1][j - 1]

        ans = 0
        for i in range(n):
            tot = i
            u = i
            for j in range(40):
                if k & (1 << j):
                    u = up[u][j]
                    tot += dp[u][j]
            ans = max(ans, tot)

        return ans

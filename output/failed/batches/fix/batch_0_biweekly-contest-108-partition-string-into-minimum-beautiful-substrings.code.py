from typing import List


class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        pow_bank = set()
        p = 1
        while p < (1 << len(s)):
            pow_bank.add(p)
            p *= 5
        dp = [float('inf')] * (len(s) + 1)
        dp[len(s)] = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '0':
                continue
            for j in range(i, len(s)):
                dec = int(s[i:j + 1], 2)
                if dec in pow_bank:
                    dp[i] = min(dp[i], dp[j] + 1)
        if dp[0] == float('inf'):
            return -1
        return dp[0]

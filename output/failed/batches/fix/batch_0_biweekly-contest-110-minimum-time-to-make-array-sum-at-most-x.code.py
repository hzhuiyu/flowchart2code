from typing import List


class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        s = sum(nums1)
        d = sum(nums2)
        if s <= x:
            return 0
        pairs = sorted(zip(nums2, nums1))
        dp = [0] * (n + 1)
        for k in range(1, n + 1):
            b, a = pairs[k - 1]
            for j in range(1, k + 1):
                cand = dp[j - 1] + b * j + a
                if cand > dp[j]:
                    dp[j] = cand
        for t in range(1, n + 1):
            if s + t * d - dp[t] <= x:
                return t
        return -1

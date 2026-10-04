from typing import List


class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        IDX = [0] * 32
        ans = [0] * n
        for x in nums:
            for i in range(32):
                if (x >> i) & 1:
                    ans[IDX[i]] += 1 << i
                    IDX[i] += 1
        mod = 10 ** 9 + 7
        res = 0
        for i in range(k - 1):
            res += ans[i] ** 2
            res %= mod
        return res

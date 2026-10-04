class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        m = k // 2
        if n <= m:
            return n * (n + 1) // 2
        total = m * (m + 1) // 2
        cnt = n - m
        total += cnt * k + cnt * (cnt + 1) // 2
        return total

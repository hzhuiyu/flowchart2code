from typing import List


class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
        res = float('inf')
        S = []
        for i in range(len(nums)):
            if i >= x - 1:
                S.append(nums[i - x + 1])
                S = sorted(S)
            t = nums[i]
            lo, hi = 0, len(S)
            while lo < hi:
                mid = (lo + hi) // 2
                if S[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo
            if j < len(S):
                res = min(res, abs(S[j] - nums[i]))
            if j > 0:
                res = min(res, abs(S[j - 1] - nums[i]))
        return int(res)

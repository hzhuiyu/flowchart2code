from typing import List


class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        i = 0
        while i < n:
            j = i
            while j < n:
                ok = 1
                lst = -1
                k = 0
                while k < n:
                    if i < k <= j:
                        k += 1
                    elif lst < nums[k]:
                        lst = nums[k]
                        k += 1
                    else:
                        ok = 0
                        k += 1
                if ok == 1:
                    ans += 1
                j += 1
            i += 1
        return ans

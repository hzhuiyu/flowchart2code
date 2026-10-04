from typing import List


class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        r = 0
        i = 0
        while i < n:
            # s holds the distinct values of nums[i..j-1]; temp is the
            # imbalance number of the current subarray nums[i..j].
            s = set()
            temp = 0
            j = i
            while j < n:
                x = nums[j]
                if x not in s:
                    temp += 1 - ((x - 1) in s) - ((x + 1) in s)
                    s.add(x)
                r += temp
                j += 1
            i += 1
        return r

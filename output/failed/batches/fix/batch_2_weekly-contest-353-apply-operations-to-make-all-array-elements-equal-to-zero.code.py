from typing import List


class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # arr[i] records how much the active decrement count drops at index i
        # (operations started at i-k end there). h is the current decrement
        # level applied to position i.
        arr = [0] * (n + 1)
        h = 0
        for i in range(n):
            v = nums[i]
            h -= arr[i]
            if v > h:
                # Need v - h more operations starting at i; they must fit.
                if i + k > n:
                    return False
                arr[i + k] = v
                h = v
            elif v < h:
                return False
        return True

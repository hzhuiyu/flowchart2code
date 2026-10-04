from typing import List


class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        # Two pointers with monotonic deques holding indices of the window's
        # max (decreasing) and min (increasing) candidates.
        maxq = []  # indices, nums values decreasing
        minq = []  # indices, nums values increasing
        l = 0
        count = 0
        for r in range(len(nums)):
            v = nums[r]
            while maxq and nums[maxq[-1]] <= v:
                maxq.pop()
            maxq.append(r)
            while minq and nums[minq[-1]] >= v:
                minq.pop()
            minq.append(r)
            # Shrink from the left while the window violates |max - min| <= 2.
            while nums[maxq[0]] - nums[minq[0]] >= 2:
                l += 1
                if maxq[0] < l:
                    maxq.pop(0)
                if minq[0] < l:
                    minq.pop(0)
            count += r - l + 1
        return count

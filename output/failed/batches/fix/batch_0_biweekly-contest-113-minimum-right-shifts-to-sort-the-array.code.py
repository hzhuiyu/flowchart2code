from typing import List


class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = 0
        desc = -1
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                cnt += 1
                desc = i
        if cnt == 0:
            return 0
        if cnt == 1:
            return n - desc
        return -1

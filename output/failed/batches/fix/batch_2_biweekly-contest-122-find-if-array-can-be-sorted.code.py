from typing import List


class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Popcount of every element.
        a = []
        for v in nums:
            c = 0
            w = v
            while w:
                c += w & 1
                w >>= 1
            a.append(c)

        n = len(nums)
        ok = True
        prev_max = -1
        i = 0
        # Elements with equal popcount can be freely rearranged inside each
        # consecutive group; groups cannot mix. Sortable iff the previous
        # group's maximum does not exceed the next group's minimum.
        while i < n:
            cnt = a[i]
            j = i
            while j < n and a[j] == cnt:
                j += 1
            gmin = min(nums[i:j])
            gmax = max(nums[i:j])
            if prev_max > gmax:
                ok = False
            prev_max = gmax
            i = j
        return ok

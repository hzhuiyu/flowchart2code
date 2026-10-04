from typing import List


class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        res = 0
        i = 20
        while i > 0:
            res <<= 1
            pref = {}
            pref2 = {}
            j = 0
            while j < len(nums):
                a = nums[j]
                p = a >> i
                if p in pref:
                    pref2[p] = max(pref2[p], a)
                    pref[p] = min(pref[p], a)
                else:
                    pref[p] = a
                    pref2[p] = a
                j += 1
            ks = list(pref.keys())
            kx = 0
            while kx < len(ks):
                x = ks[kx]
                y = res ^ 1 ^ x
                if x >= y and y in pref:
                    if pref[x] <= pref2[y] * 2:
                        res |= 1
                kx += 1
            i -= 1
        return res

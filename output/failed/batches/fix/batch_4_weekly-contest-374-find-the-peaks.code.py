from typing import List


class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        res = []
        i = 1
        n = len(mountain) - 2

        while i < n:
            if mountain[i] > mountain[i - 1]:
                if mountain[i] > mountain[i + 1]:
                    res.append(i)
            i += 1

        return res

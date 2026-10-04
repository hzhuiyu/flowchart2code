from typing import List


class Solution:
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: List[int], vBars: List[int]) -> int:
        hBars.sort()
        vBars.sort()

        hmax = 0
        run = 0
        for i in range(len(hBars)):
            hmax = max(hmax, run)
            if i > 0 and hBars[i] == hBars[i - 1] + 1:
                run += 1
            else:
                run = 1
        hside = hmax + 1

        gmax = 0
        run = 0
        for i in range(len(vBars)):
            if i > 0 and vBars[i] == vBars[i - 1] + 1:
                run += 1
            else:
                run = 1
            gmax = max(gmax, run)
        gside = gmax + 1

        return min(hside, gside) ** 2

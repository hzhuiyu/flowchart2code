from typing import List


class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        fre = [0] * (n * n + 1)
        rep = 0
        mis = 0

        for i in range(n):
            for num in grid[i]:
                fre[num] += 1

        for j in range(1, n * n):
            if fre[j] == 0:
                mis = j
            elif fre[j] == 2:
                rep = j

        return [rep, mis]

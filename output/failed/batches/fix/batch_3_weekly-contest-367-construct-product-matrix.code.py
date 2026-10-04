from typing import List


class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        n = len(grid)
        m = len(grid[0])
        flat = [v for row in grid for v in row]
        nm = len(flat)
        suf = [1] * (nm + 1)
        for i in range(nm - 1, -1, -1):
            suf[i] = suf[i + 1] * flat[i] % mod
        pre = 1
        for i in range(nm):
            grid[i // m][i % m] = pre * suf[i] % mod
            pre = pre * flat[i] % mod
        return grid

from typing import List


class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        temp = [(0, 0)]
        for i in range(n):
            pos = 2 * i + 1
            new = temp[0][1] + prices[i]
            if i == temp[0][0]:
                temp.pop(0)
            while temp and temp[-1][1] >= new:
                temp.pop()
            temp.append((pos, new))
        return temp[0][1]

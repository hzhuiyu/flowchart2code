from typing import List


class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins = sorted(coins)
        i = 0
        max_sofar = 0
        ans = 0
        while max_sofar < target:
            if i < len(coins) and coins[i] <= max_sofar:
                max_sofar += coins[i]
                i += 1
            else:
                ans += 1
                max_sofar += max_sofar + 1
        return ans

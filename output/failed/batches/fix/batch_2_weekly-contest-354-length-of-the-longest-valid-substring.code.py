from typing import List


class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        res = 0
        # right is the furthest index a valid substring ending at left may use;
        # any forbidden substring found while scanning right-to-left shrinks it.
        right = len(word) - 1
        left = len(word) - 1
        while left >= 0:
            k = left
            while k < min(left + 10, right + 1):
                if word[left:k + 1] in forbidden_set:
                    right = k
                    break
                k += 1
            res = max(res, right - left + 1)
            left -= 1
        return res

from typing import List


class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        # Sort by profit descending; greedily build the top-k selection,
        # then exchange duplicate-category items for new-category ones.
        items = sorted(items, key=lambda x: x[0], reverse=True)
        total = 0
        seen = set()
        dups = []  # profits of selected items whose category is duplicated
        ans = 0
        idx = 0
        while idx < len(items):
            p = items[idx][0]
            c = items[idx][1]
            if idx < k:
                total += p
                if c in seen:
                    dups.append(p)
                else:
                    seen.add(c)
            else:
                if len(dups) > 0 and c not in seen:
                    total += dups.pop() - p
                    seen.add(c)
            cand = total + len(seen) * len(seen)
            ans = max(ans, cand)
            idx += 1
        return ans

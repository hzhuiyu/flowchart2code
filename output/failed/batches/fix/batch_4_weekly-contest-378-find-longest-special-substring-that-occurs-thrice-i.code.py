class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        best = -1
        for c in set(s):
            runs = []
            cnt = 0
            for ch in s:
                if ch == c:
                    cnt += 1
                else:
                    if cnt > 0:
                        runs.append(cnt)
                    cnt = 0
            if cnt > 0:
                runs.append(cnt)
            for L in range(1, max(runs) + 1):
                occ = 0
                for r in runs:
                    if r >= L:
                        occ += r - L
                if occ >= 3 and L > best:
                    best = L
        return best

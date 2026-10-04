class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        s = moves.count('L')
        t = moves.count('R')
        u = moves.count('_')
        return abs(t - s) - u

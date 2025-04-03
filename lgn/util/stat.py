class Cached:
    SOLVER = "solver"


class Stat:
    cache_hit = {Cached.SOLVER: 0}
    cache_miss = {Cached.SOLVER: 0}

    def inc_cache_hit(flag: str):
        Stat.cache_hit[flag] += 1

    def inc_cache_miss(flag: str):
        Stat.cache_miss[flag] += 1

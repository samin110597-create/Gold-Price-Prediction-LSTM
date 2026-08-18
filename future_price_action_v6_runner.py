import numpy as np
import future_price_action_v6 as core


def maturity_safe_origins(n,h,max_origins=72):
    """Origins are spaced at least one full forecast horizon apart.
    This guarantees that every prior OOS outcome used for online model weighting,
    probability calibration, and residual bands would have been observable before
    the next forecast origin.
    """
    min_train=1000
    spacing=8 if h==1 else (10 if h==5 else max(21,h+1))
    end=n-h-1
    if end<=min_train+h:return []
    pool=np.arange(min_train+h,end+1,spacing)
    return (pool[-max_origins:] if len(pool)>max_origins else pool).tolist()

core.origin_positions=maturity_safe_origins

if __name__=='__main__':
    core.main()

import QgyModel
from SwapPricerByQgy import *
import numpy as np


class IICapFloorQgy(QgyModel):
    def price_caplet_by_qgy(self, k, T, K, P_0T):
        swaplet_pricer = IISwapQGY()
        E0_DY = swaplet_pricer.price_swaplet_by_qgy(k-1, k, T)
        ND1 = self.compute_Nd()
        ND2 = self.compute_Nd()
        return E0_DY * ND1 - K * P_0T * ND2

    def compute_Nd(self):
        

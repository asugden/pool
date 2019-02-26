import flow.trace2p

if __name__ == '__main__':
    t2p = flow.trace2p.Trace2P('/Users/arthur/Documents/Programs/replay-classifier/1-data/CB173/160513/CB173_160513_002.simpcell')
    temp = t2p.warpcstraces('plus', end_s=4)
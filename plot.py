import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
path = '/Users/neythen/Desktop/Projects/OED_toggle_switch/results'

for i in range(1, 11):

    print()
    print(i)

    try:
        returns = np.load(path + '/repeat' + str(i) + '/all_returns.npy') * 100
        print(returns)
        print('end:', returns[-1])
        print('max:', np.max(returns))
    except:
        pass


    plt.figure()
    plt.plot(returns)

    actions = np.load(path + '/repeat' + str(i) + '/actions.npy')


    n_unstable = np.load(path + '/repeat' + str(i) + '/n_unstables.npy')


plt.show()
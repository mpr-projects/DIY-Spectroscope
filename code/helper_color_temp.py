import numpy as np


def get_radiance(T, nms, refractive_index=1, normalize=True):
    # temp is in Kelvin
    m = nms * 1e-9
    h =  6.62607015e-34  # J/Hz
    c = 299792458        # m/s
    kB = 1.380649e-23    # J/K

    # value 1 is in vauum, other media require adjustment
    c /= refractive_index

    res = 2 * h * c**2 / m**5
    res /= np.exp(h * c / (m * kB * T)) - 1

    if normalize is True:
        res = res / np.amax(res)
    
    return res



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nms = np.arange(2, 2000)

    for T in [2500, 2700, 3000, 4000]:
        rad = get_radiance(T, nms, normalize=False)
        plt.plot(nms, rad, label=f'{T}K')

    plt.legend()
    plt.show()

    # compare result for different refractive indices 
    rad = get_radiance(3000, nms, refractive_index=1, normalize=False)
    plt.plot(nms, rad, label='vacuum')

    rad = get_radiance(3000, nms, refractive_index=1.000277, normalize=False)
    plt.plot(nms, rad, label='air')

    plt.legend()
    plt.show()

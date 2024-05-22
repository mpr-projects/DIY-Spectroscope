import jax.numpy as np


def get_radiance(T, nms, refractive_index=1, normalize=True):
    # temp is in Kelvin
    T = np.atleast_1d(T)

    m = nms  # rescaled values
    h =  6.62607015
    c = 2.99792458
    kB = 1.380649

    # value 1 is in vacuum, other media require adjustment
    c /= refractive_index

    m, T = m[None, :], T[:, None]

    num = 2 * h * c**2 / m**5 * 1e+27
    den = np.exp(1e6 * h * c / (m * kB * T)) - 1
    res = num / den

    if normalize is True:
        res = res / np.amax(res, axis=1, keepdims=True)

    return res

def get_radiance_(T, nms, refractive_index=1, normalize=True):
    print('inputs:', T.shape, nms.shape)
    # temp is in Kelvin
    T = np.atleast_1d(T)

    m = nms * 1e-9
    h =  6.62607015e-34  # J/Hz
    c = 299792458        # m/s
    kB = 1.380649e-23    # J/K

    # value 1 is in vacuum, other media require adjustment
    c /= refractive_index

    m, T = m[None, :], T[:, None]

    num = 2 * h * c**2 / m**5
    den = np.exp(h * c / (m * kB * T)) - 1
    res = num / den 

    if normalize is True:
        res = res / np.amax(res, axis=1, keepdims=True)

    return res



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nms = np.arange(2, 2000)

    Ts = np.array([2500, 2700, 3000, 4000])

    rads = get_radiance(Ts, nms, normalize=True)

    for rad, T in zip(rads, Ts):
        plt.plot(nms, rad, label=f'{T}K')

    plt.legend()
    plt.show()

    for T in Ts:
        rad = get_radiance(T, nms, normalize=False)[0]
        plt.plot(nms, rad, label=f'{T}K')

    plt.legend()
    plt.show()

    # compare result for different refractive indices 
    rad = get_radiance(3000, nms, refractive_index=1, normalize=False)[0]
    plt.plot(nms, rad, label='vacuum')

    rad = get_radiance(3000, nms, refractive_index=1.000277, normalize=False)[0]
    plt.plot(nms, rad, label='air')

    plt.legend()
    plt.show()

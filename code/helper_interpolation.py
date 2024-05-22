import numpy as np
from scipy.interpolate import CubicSpline


def pw_linear(xs, ys, xt, y_axis=None):
    xs = np.array(xs)
    ys = np.array(ys)
    xt = np.array(xt)

    is_larger = (xt[:, None] > xs[None, :-1])
    is_smaller = (xt[:, None] <= xs[None, 1:])

    a = np.logical_and(is_larger, is_smaller)

    # account for extrapolation
    extrap_idx = np.zeros(len(xs)-1, dtype=bool)
    extrap_idx[0] = True  # lower end

    idxs = np.logical_not(np.any(is_larger, axis=-1))
    a[idxs] = extrap_idx

    extrap_idx = np.zeros(len(xs)-1, dtype=bool)
    extrap_idx[-1] = True  # upper end

    idxs = np.logical_not(np.any(is_smaller, axis=-1))
    a[idxs] = extrap_idx

    # find interval indices
    idx = np.where(a == True)[1]

    # interpolate
    x0 = np.take(xs, idx)
    y0 = np.take(ys, idx, axis=y_axis)
    x1 = np.take(xs, idx+1)
    y1 = np.take(ys, idx+1, axis=y_axis)

    yt = y0 + (xt-x0)/(x1-x0)*(y1-y0)
    return yt



# from https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html
def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])


def cubic_spline(xs, ys, xt):
    cs = CubicSpline(xs, ys, bc_type='natural')
    add_boundary_knots(cs)
    return cs(xt)


"""
from scipy.interpolate import BSpline, make_interp_spline
sp = make_interp_spline(px, nm, k=1)  #  bc_type='natural',
nms = sp(xvals)
# """

# nms = np.interp(xvals, px, nm)
# nms = (w1*p2-p1*w2)/(p2-p1) + xvals * (w2-w1)/(p2-p1)



mapping = {
    'Piecewise Linear': pw_linear,
    'Cubic Spline': cubic_spline
}

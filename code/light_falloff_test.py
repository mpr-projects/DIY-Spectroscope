import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from scipy.optimize import curve_fit
import numpy as np
import rawpy
import copy
import os

from helper_misc import px, is_float
from model_image import Image


class Window(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Light Fall-Off Test')

        self.cos_power = 4

        self.bind('<Control-q>', self.quit)
        self.bind('<Control-l>', self.load_image)

        # Temp
        self.bind('<Control-a>', self.animate)

        self.raw = None

        # put all subframes into a ttk.Frame
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.f = f = ttk.Frame(self)
        f.grid(row=0, column=0, sticky='nesw')

        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        # plot 1: image
        self.fig1 = fig = Figure()
        self.ax1 = ax = fig.add_subplot()

        self.canvas1 = canvas = FigureCanvasTkAgg(fig, master=f)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, f, pack_toolbar=False)
        toolbar.update()
        self.toolbar1 = toolbar

        canvas.mpl_connect("key_press_event", key_press_handler)

        canvas.get_tk_widget().grid(row=0, column=0, sticky='nesw')
        toolbar.grid(row=1, column=0, sticky='ns')

        # plot 2: info
        self.fig2 = fig = Figure()
        self.ax2 = ax = fig.add_subplot()

        self.canvas2 = canvas = FigureCanvasTkAgg(fig, master=f)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, f, pack_toolbar=False)
        toolbar.update()
        self.toolbar1 = toolbar

        canvas.mpl_connect("key_press_event", key_press_handler)

        canvas.get_tk_widget().grid(row=0, column=1, sticky='nesw')
        toolbar.grid(row=1, column=1, sticky='ns')

        # sidebar
        sb_frame = ttk.Frame(f)
        sb_frame.grid(
            row=0, column=2, rowspan=2, sticky='nesw', padx=10, pady=10)

        self.build_gui(sb_frame)

    def build_gui(self, f):
        b = ttk.Button(f, text='Load Image', underline=0,
               command=self.load_image).grid(row=0, column=0)

        # cropping
        ttk.Label(f, text='Crop Image').grid(
            row=1, column=0, pady=(20, 5))

        f_ = ttk.Frame(f)
        f_.grid(row=2, column=0)

        ttk.Label(f_, text='X').grid(row=0, column=0, padx=(10,5))

        def validate(newval):
            return newval.isdigit()

        validate = (f.register(validate), '%P')

        self.var_startx = tk.IntVar()
        self.txt_crop_start_x = e = ttk.Entry(
            f_, textvariable=self.var_startx, width=4,
            validate='key', validatecommand=validate)
        e.grid(row=0, column=1)
        e.bind('<Return>', self.crop_image)

        ttk.Label(f_, text='-').grid(row=0, column=2, padx=5)

        self.var_endx = tk.IntVar()
        e = ttk.Entry(f_, textvariable=self.var_endx, width=4,
                      validate='key', validatecommand=validate)
        e.grid(row=0, column=3, padx=(0, 10))
        e.bind('<Return>', self.crop_image)

        ttk.Label(f_, text='Y').grid(row=1, column=0, padx=(10,5))

        self.var_starty = tk.IntVar()
        e = ttk.Entry(f_, textvariable=self.var_starty, width=4,
                      validate='key', validatecommand=validate)
        e.grid(row=1, column=1)
        e.bind('<Return>', self.crop_image)

        ttk.Label(f_, text='-').grid(row=1, column=2, padx=5)

        self.var_endy = tk.IntVar()
        e = ttk.Entry(f_, textvariable=self.var_endy, width=4,
                      validate='key', validatecommand=validate)
        e.grid(row=1, column=3, padx=(0, 10))
        e.bind('<Return>', self.crop_image)

        # choose color
        f_ = ttk.Frame(f)
        f_.grid(row=3, column=0, pady=(15, 0))

        self.var_color = v = tk.IntVar(value=1)
        ttk.Radiobutton(
            f_, text='Show Red', variable=v, value=0, command=self.update_color
        ).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(
            f_, text='Show Green', variable=v, value=1, command=self.update_color
        ).grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(
            f_, text='Show Blue', variable=v, value=2, command=self.update_color
        ).grid(row=2, column=0, sticky='w')

        # x-coord slider
        ttk.Label(f, text='Choose X-Slice').grid(
            row=13, column=0, pady=(20, 5))

        self.scale_x = ttk.Scale(
            f, orient='horizontal', length=100,
            command=self.changed_x_slice)
        self.scale_x.grid(row=14, column=0, padx=10)

        # correction
        ttk.Label(f, text='Correction Settings').grid(
            row=15, column=0, pady=(20, 5))

        f_ = ttk.Frame(f)
        f_.grid(row=16, column=0)

        ttk.Label(f_, text='x-center').grid(
            row=0, column=0, padx=(10, 5), sticky='w')

        self.var_center_x = tk.IntVar()
        e = ttk.Entry(f_, textvariable=self.var_center_x, width=4,
                      validate='key', validatecommand=validate)
        e.grid(row=0, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot1)

        self.var_link_slice_and_center = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f_, variable=self.var_link_slice_and_center,
            text='Link to X-Slice'
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 0))

        # cos^4 parameter fit v2
        validate_float = (f.register(is_float), '%P')

        ttk.Label(f, text='Cos4 Parameter Fit').grid(
            row=29, column=0, pady=(20, 5))

        f_ = ttk.Frame(f)
        f_.grid(row=30, column=0)

        ttk.Label(f_, text='y0').grid(
            row=0, column=0, padx=(10, 5), sticky='w')

        self.var_fit_y0 = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_y0, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=0, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='v0').grid(
            row=1, column=0, padx=(10, 5), sticky='w')

        self.var_fit_v0 = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_v0, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=1, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='y1').grid(
            row=2, column=0, padx=(10, 5), sticky='w')

        self.var_fit_y1 = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_y1, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=2, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='y2').grid(
            row=3, column=0, padx=(10, 5), sticky='w')

        self.var_fit_y2 = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_y2, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=3, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='a').grid(
            row=4, column=0, padx=(10, 5), sticky='w')

        self.var_fit_a = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_a, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=4, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='c').grid(
            row=5, column=0, padx=(10, 5), sticky='w')

        self.var_fit_c = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_c, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=5, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='yb').grid(
            row=6, column=0, padx=(10, 5), sticky='w')

        self.var_fit_yb = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_yb, width=8,
                      validate='key', validatecommand=validate)
        e.grid(row=6, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        ttk.Label(f_, text='ye').grid(
            row=7, column=0, padx=(10, 5), sticky='w')

        self.var_fit_ye = tk.DoubleVar()
        e = ttk.Entry(f_, textvariable=self.var_fit_ye, width=8,
                      validate='key', validatecommand=validate_float)
        e.grid(row=7, column=1, padx=(0, 10))
        e.bind('<Return>', self.plot2)

        self.var_fit_linear = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f_, variable=self.var_fit_linear,
            text='Linear Extrapolation'
        ).grid(row=8, column=0, columnspan=2, padx=10, pady=(9, 0))

        f_ = ttk.Frame(f)
        f_.grid(row=31, column=0)

        ttk.Button(f_, text='Guess', command=self.guess_cos4).grid(
            row=0, column=0, pady=(9, 0), padx=5)

        ttk.Button(f_, text='Fit', command=self.fit_cos4).grid(
            row=0, column=1, pady=(9, 0))

        ttk.Label(f, text='Radial Adjustment').grid(
            row=32, column=0, pady=(20, 5))

        f_ = ttk.Frame(f)
        f_.grid(row=33, column=0)

        ttk.Label(f_, text='Black Level').grid(
            row=0, column=0, padx=(10, 5), sticky='w')

        self.var_adj_blg = tk.IntVar()
        e = ttk.Entry(f_, textvariable=self.var_adj_blg, width=4,
                      validate='key', validatecommand=validate)
        e.grid(row=0, column=1, padx=(0, 10), sticky='w')
        # e.bind('<Return>', self.evaluate_composite_fn)

        ttk.Label(f_, text='Method').grid(
            row=1, column=0, padx=(10, 5), pady=(3, 0), sticky='w')

        self.var_adj_method = tk.StringVar(value='vertical')

        ttk.Radiobutton(
            f_, text='vertical', variable=self.var_adj_method,
            value='vertical').grid(row=1, column=1, sticky='w')

        ttk.Radiobutton(
            f_, text='radial', variable=self.var_adj_method,
            value='radial').grid(row=2, column=1, pady=(3, 0), sticky='w')

        f_ = ttk.Frame(f)
        f_.grid(row=34, column=0)

        ttk.Button(
            f_, text='Guess', command=self.guess_black_level
        ).grid(row=0, column=0, padx=5, pady=(9, 0))

        ttk.Button(
            f_, text='Apply', command=self.apply_adjustment
        ).grid(row=0, column=1, pady=(9, 0))

        ttk.Button(
            f_, text='Reset', command=self.reset
        ).grid(row=1, column=0, columnspan=2, pady=(5, 0))

    @property
    def center_x(self):
        return px(self.var_center_x.get())

    def quit(self, *args):
        self.destroy()

    def load_image(self, *args):
        fpath = filedialog.askopenfilename()

        if len(fpath) == 0:
            return

        # could use this as a standalone script with the line below,
        # (would also have to take care of the two helper imports)
        # self.raw = rawpy.imread(fpath)

        # or use the Image model for lens corrections
        self.calibration = lambda: None
        # self.calibration.lens_corrections = [True, False, True]
        # self.calibration.black_levels = [1050, 1050, 1050]
        self.calibration.lens_corrections = [False, False, False]
        self.raw = Image(self, fpath)

        # only keep green values (most pixels are green + green has
        # the strongest signal in the center of the image),
        # self._vals contains the original data, self.vals the cropped data
        cols = self.raw.raw_colors
        vals = self.raw.raw_image

        self._vals = np.where(cols==1, vals, -1)
        self.vals = copy.deepcopy(self._vals)
        self.start_x = 0
        self.start_y = 0

        self.update_scale_x()
        self.plot()
        self.txt_crop_start_x.focus_set()

        # return
        # Temp
        self.var_startx.set(2000)
        self.var_endx.set(4200)
        self.var_starty.set(1000)
        self.var_endy.set(2400)
        self.crop_image()
        self.scale_x.set(3173)
        self.plot()

    def crop_image(self, *args):
        if self.raw is None:
            return

        print('cropping image')

        self.start_x = self.var_startx.get() or 0
        self.start_y = self.var_starty.get() or 0

        h, w = self._vals.shape

        end_x = self.var_endx.get()
        end_y = self.var_endy.get()

        if end_x == 0:
            end_x = w

        if end_y == 0:
            end_y = h

        if self.start_x > end_x:
            return

        if self.start_y > end_y:
            return

        self.vals = copy.deepcopy(
            self._vals[self.start_y:end_y, self.start_x:end_x])

        self.update_scale_x()
        self.plot()

    def update_color(self, *args, cid=None):
        if cid is None:
            cid = self.var_color.get()

        cols = self.raw.raw_colors
        vals = self.raw.raw_image

        self._vals = np.where(cols==cid, vals, -1)
        self.vals = copy.deepcopy(self._vals)
        self.crop_image()

        if getattr(self, '_applied_adjustment', False) is True:
            self._applied_adjustment = False
            self.apply_adjustment()

    def update_scale_x(self):
        if self.raw is None:
            return

        val = self.scale_x.get()

        h, w = self.vals.shape
        f, t = self.start_x, self.start_x+w

        if val < f or val > t:
            val = (f + t) / 2

        self.scale_x.configure(from_=f, to=t, value=val)

    def changed_x_slice(self, *args):
        link = self.var_link_slice_and_center.get()

        if link is True:
            self.var_center_x.set(self.scale_x.get())

        self.plot()

    def _fn_cos4(self, y, a, b, c, d):
        return a + b * np.cos(c * (y - d))**self.cos_power

    def _fn_d_cos4(self, y, a, b, c, d):
        return -self.cos_power*b*c*np.sin(c*(y-d))*np.cos(c*(y-d))**(self.cos_power-1)

    def guess_cos4(self):
        if self.raw is None:
            return

        ys, vals, x = self.get_y_vals(x='center')

        mask = (vals > 0)
        ys = ys[mask]
        vals = vals[mask]

        # guess left and right boundary of data from gradient,
        # adding some offset to avoid the boundary effects
        grad = (vals[1:] - vals[:-1]) / (ys[1:] - ys[:-1])
        idx_yb, idx_ye = np.argmax(grad), np.argmin(grad)
        yb, ye = ys[idx_yb] + 50, ys[idx_ye] - 50
        idx_yb = np.argmin(np.abs(ys-yb))
        idx_ye = np.argmin(np.abs(ys-ye))
        yb, ye = ys[idx_yb], ys[idx_ye]
        vb = vals[idx_yb]

        self.var_fit_yb.set(yb)
        self.var_fit_ye.set(ye)

        max_idx = np.argmax(vals)
        y_max = ys[max_idx]
        self.var_fit_y0.set(y_max)

        v_max = vals[max_idx]
        self.var_fit_v0.set(v_max)

        # not sure how to best guess y1, just making it very
        # similar to y0
        y1 = y_max - 20
        self.var_fit_y1.set(y1)

        self.var_fit_y2.set((yb+y1)/2) 

        # choose a and c such that the middle of the downward sloping
        # cosine is at the left edge of the signal
        self.var_fit_a.set(4/3*vb - v_max/3)
        self.var_fit_c.set(np.pi/4/(yb-y1))

        self.has_cos4_fit = True
        self.plot2()

    def _composite_fn(self, ys, y0, v0, y1, y2, a, c):
        # need to sort the values, otherwise during optimization the
        # order may not be maintained (resulting in a differently shaped
        # function
        y2, y1, y0 = sorted([y2, y1, y0])

        # need to adjust the y-coordinates for the right side of the function
        y1_ = y0 + (y0 - y1)
        y2_ = 2*y0 - y2

        b, d = v0-a, y1

        res = np.zeros_like(ys)

        mask_1 = np.abs(ys-y0) <= np.abs(y1-y0)
        res[mask_1] = v0

        if self.var_fit_linear.get() is True:
            mask_2 = (ys <= y2)
            fy2 = self._fn_cos4(y2, a, b, c, d=y1)
            dfy2 = self._fn_d_cos4(y2, a, b, c, d=y1)
            res[mask_2] = fy2 + dfy2*(ys[mask_2] - y2)

            mask_3 = ys >= y2_
            fy2 = self._fn_cos4(y2_, a, b, c, d=y1_)
            dfy2 = self._fn_d_cos4(y2_, a, b, c, d=y1_)
            res[mask_3] = fy2 + dfy2*(ys[mask_3] - y2_)

            mask = np.logical_or(mask_1, mask_2)
            mask = np.logical_or(mask, mask_3)
            mask = np.logical_not(mask)

        else:
            mask = np.logical_not(mask_1)

        mask_4 = np.logical_and(mask, ys < y0)
        res[mask_4] = self._fn_cos4(ys[mask_4], a, b, c, d=y1)

        mask_5 = np.logical_and(mask, ys > y0)
        res[mask_5] = self._fn_cos4(ys[mask_5], a, b, c, d=y1_)

        return res

    def evaluate_composite_fn(self, ys):
        if self.raw is None:
            return

        y0 = self.var_fit_y0.get()
        v0 = self.var_fit_v0.get()
        y1 = self.var_fit_y1.get()
        y2 = self.var_fit_y2.get()
        a = self.var_fit_a.get()
        c = self.var_fit_c.get()

        return self._composite_fn(ys, y0, v0, y1, y2, a, c)

    def fit_cos4(self):
        if self.raw is None:
            return

        y0 = self.var_fit_y0.get()
        v0 = self.var_fit_v0.get()
        y1 = self.var_fit_y1.get()
        y2 = self.var_fit_y2.get()
        a = self.var_fit_a.get()
        c = self.var_fit_c.get()

        # initial guess
        p0 = (y0, v0, y1, y2, a, c)

        def fn(ys, *args):
            return self._composite_fn(ys, *args)

        yb = self.var_fit_yb.get()
        ye = self.var_fit_ye.get()

        ys, vals, _ = self.get_y_vals(x='center')

        mask = (vals > 0)
        ys, vals = ys[mask], vals[mask]

        mask = np.logical_and(ys >= yb, ys <= ye)
        ys, vals = ys[mask], vals[mask]

        try:
            popt, pcov = curve_fit(
                fn, ys, vals, p0=p0, method='lm', maxfev=10000)
        except RuntimeError:
            messagebox.showerror(title='Error', message=(
                'Optimization failed. This can happen. Try again with'
                ' a slighly different parameter guess.'))
            return

        self.var_fit_y0.set(popt[0])
        self.var_fit_v0.set(popt[1])
        self.var_fit_y1.set(popt[2])
        self.var_fit_y2.set(popt[3])
        self.var_fit_a.set(popt[4])
        self.var_fit_c.set(popt[5])

        self.has_cos4_fit = True
        self.var_link_slice_and_center.set(False)
        self.plot2()

    def guess_black_level(self):
        if self.raw is None or self.has_cos4_fit is False:
            return

        yb, ye = px(self.var_fit_yb.get(), self.var_fit_ye.get())
        yb, ye = yb - 100, ye + 100
        x = self.center_x

        if x == 0:
            return

        vals = self._vals[:, x]
        # vals[yb:ye] = -1
        vals = np.concatenate((vals[:yb], vals[ye:]))

        # there's always some noise, the only points where the value
        # is exactly zero is where rawpy adds some padding
        vals = vals[vals > 0]

        bl = int(round(np.mean(vals)))
        self.var_adj_blg.set(bl)

    def compute_radial_adjustment(self, max_r):
        # max_r represents the maximum distance from y0 in pixels,
        # this entire function uses pixels, could also do it in 
        # floats and round to pixels later but this is easier to
        # implement
        if self.raw is None or self.has_cos4_fit is False:
            return

        y0 = px(self.var_fit_y0.get())
        ys = np.arange(y0, y0+max_r+1)
        vals = self.evaluate_composite_fn(ys)

        bl = self.var_adj_blg.get()
        return (vals[0] - bl) / (vals - bl)

    def apply_radial_adjustment(self):
        print('in apply_radial_adjustment')
        if self.raw is None or self.has_cos4_fit is False:
            return

        h, w = self.vals.shape
        start_x, start_y = self.start_x, self.start_y
        end_x, end_y = start_x + w, start_y + h

        tl = np.array([start_x, start_y])
        tr = np.array([end_x, start_y])
        br = np.array([end_x, end_y])
        bl = np.array([start_x, end_y])

        cc = np.array([self.center_x, self.var_fit_y0.get()])
        print('coords:', cc, tl, tr, br, bl)

        if getattr(self, '_adj_data', None) is None:
            d = lambda x, y: sum((x-y)**2)**0.5
            max_r = max([d(c, cc) for c in [tl, tr, br, bl]])

            self._adj_data = (self.compute_radial_adjustment(max_r), max_r)

        ra, max_r = self._adj_data

        """ ""
        import matplotlib.pyplot as plt
        plt.plot(ra)
        plt.show()
        # """

        ys = np.arange(start_y, end_y)
        xs = np.arange(start_x, end_x)
        XY = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis=-1)
        print('XY:', XY)

        print('shapes:', xs.shape, ys.shape, self.vals.shape, XY.shape)

        dists = XY - cc[None, None, :]
        dists = np.sum(dists**2, axis=-1)**0.5
        
        """ ""
        extent = (start_x, end_x, end_y, start_y)
        plt.imshow(dists, origin='upper', extent=extent)
        plt.show()
        # """

        rs = np.arange(max_r+1)
        print('shapes_:', rs.shape, ra.shape)
        dists_shape = dists.shape
        dists = dists.flatten()

        factors = np.interp(dists, rs, ra)
        factors = factors.reshape(dists_shape)

        """ ""
        plt.imshow(factors, origin='upper', extent=extent)
        plt.show()
        # """

        bl = self.var_adj_blg.get()
        self.vals = bl + factors * (self.vals - bl)
        self.vals = np.round(self.vals).astype(int)
        self.vals = np.where(self.vals < 0, -1, self.vals)

        self._prev_extent = None  # quick fix for plot1
        self.plot()

    def compute_vertical_adjustment(self):
        if self.raw is None or self.has_cos4_fit is False:
            return

        yb = self.var_fit_yb.get()
        ye = self.var_fit_ye.get()
        ys = np.arange(yb, ye+1)

        y0 = int(round(self.var_fit_y0.get() - yb))

        vals = self.evaluate_composite_fn(ys)

        bl = self.var_adj_blg.get()
        return ys, (vals[y0] - bl) / (vals - bl)

    def apply_vertical_adjustment(self):
        print('in apply_vertical_adjustment')
        if self.raw is None or self.has_cos4_fit is False:
            return

        if getattr(self, '_adj_data', None) is None:
            self._adj_data = self.compute_vertical_adjustment()

        ys, factors = self._adj_data

        start_y = int(ys[0]) - self.start_y
        end_y = int(ys[-1]) - self.start_y + 1

        bl = self.var_adj_blg.get()
        vals = bl + factors[:, None] * (self.vals[start_y:end_y] - bl)
        vals = np.round(vals).astype(int)
        self.vals[start_y:end_y] = np.where(vals < 0, -1, vals)

        self._prev_extent = None  # quick fix for plot1
        self.plot()

    def apply_adjustment(self):
        print('in apply_adjustment')
        is_applied = getattr(self, '_applied_adjustment', False)

        if is_applied is True:
            return

        method = self.var_adj_method.get()

        if method == 'radial':
            self.apply_radial_adjustment()

        else:
            self.apply_vertical_adjustment()

        self._applied_adjustment = True

    def reset(self):
        self._prev_extent = None  # quick fix for plot1
        self._applied_adjustment = False
        self._adj_data = None
        self.crop_image()

    def get_y_vals(self, x='center'):
        h, w = self.vals.shape

        start_x, start_y = self.start_x, self.start_y
        end_x, end_y = start_x + w, start_y + h

        if x == 'center':
            x = self.center_x

        else:
            x = px(self.scale_x.get())

        if not (x >= start_x and x <= end_x):
            x = px((start_x + end_x)/2)

        # vals = self._vals[start_y:end_y, x]
        vals = self.vals[:, x-start_x-1]
        ys = np.arange(start_y, end_y)

        return ys, vals, x

    def plot1(self, *args):
        if self.raw is None:
            return

        h, w = self.vals.shape

        extent = (self.start_x, self.start_x+w,
                  self.start_y+h, self.start_y)

        prev_extent = getattr(self, '_prev_extent', None)

        if extent == prev_extent:
            self.update1(w)
            return

        ax = self.ax1
        ax.cla()

        ax.imshow(self.vals, origin='upper', extent=extent)

        self.axvline_center_x = ax.axvline(
            self.start_x, linestyle='--', color='white', lw=0.5)

        self.axhline_center_y = ax.axhline(
            self.start_y, linestyle='--', color='white', lw=0.5)

        self.axvline_x_slice = ax.axvline(
            self.start_x, linestyle='--', color='gray', lw=0.5)

        self._prev_extent = extent
        self.update1(w)

    def update1(self, w):
        xc = self.center_x

        if xc >= self.start_x and xc <= self.start_x+w:
            self.axvline_center_x.set_xdata([xc])

        if getattr(self, 'has_cos4_fit', False) is True:
            yc = self.var_fit_y0.get()
            self.axhline_center_y.set_ydata([yc])

        xs = self.scale_x.get()

        if xs >= self.start_x and xs <= self.start_x+w:
            self.axvline_x_slice.set_xdata([xs])

        self.canvas1.draw()

    def plot2_fit(self, ax):
        h, w = self.vals.shape
        sy = self.start_y
        ey = sy + h
        ys = np.arange(sy, ey)

        vals = self.evaluate_composite_fn(ys)
        ax.plot(ys, vals)

    def plot2(self, *args):
        if self.raw is None:
            return

        ax, canvas = self.ax2, self.canvas2
        ax.cla()

        ys, vals, x = self.get_y_vals(x='current')
        start_y, end_y = ys[0], ys[-1]+1

        ax.set_title(f'X-Coordinate {x}')
        color = ['red', 'green', 'blue'][self.var_color.get()]
        ax.plot(ys, vals, 'o', ms=1, color=color)

        """
        yc = self.var_fit_y0.get()

        if yc >= start_y and yc <= end_y:
            ax.axvline(yc, linestyle='--', lw=1, color='black')

            y1 = self.var_fit_y1.get()
            yh = yc - y1

            if yh > 1e-5:
                yh = yh // 2
                ax.axvspan(yc-yh, yc+yh, alpha=0.2, color='black')
        # """

        if getattr(self, 'has_cos4_fit', False) is True:
            self.plot2_fit(ax)

        ax.set_ylim(0, self.raw.white_level)
        ax.set_xlabel('y-coordinate')
        ax.set_ylabel('measured value')
        canvas.draw()

    def plot(self, *args):
        if self.raw is None:
            return

        self.plot1()
        self.plot2()

    def animate(self, *args):
        if self.raw is None:
            return

        name = tk.simpledialog.askstring('File Name', 'Please enter a filename:')
        print('name:', name)

        if name is None:
            print('returning')
            return

        """
        print('asking for step')
        step = tk.simpledialog.askinteger('Step', 'Please enter the step size:')

        if step is None:
            return

        fr = tk.simpledialog.askinteger('Framerate', 'Please enter the framerate:')

        if fr is None:
            return
        """
        step = 1
        fr = 15

        h, w = self.vals.shape
        yb, ye = self.var_fit_yb.get(), self.var_fit_ye.get()

        if yb == 0 and ye == 0:
            yb = self.start_y
            ye = self.start_y + h

        # start_x, end_x = self.start_x, self.start_x+w+1
        start_x, end_x = 2200, 4001

        for idx, x in enumerate(range(start_x, end_x, step)):
            self.scale_x.set(x)
            self.ax2.set_xlim(yb, ye)
            self.fig2.savefig(f'output/{name}_{idx}.png',
                              bbox_inches='tight', dpi=300)

        os.system(
            f'ffmpeg -f image2 -framerate {fr}'
            f' -i output/{name}_%d.png  output/{name}.avi')

        """
        for idx in range(idx+1):
            os.remove(f'output/{name}_{idx}.png')
        """

        self.ax2.set_xlim(auto=True)
        tk.messagebox.showinfo('Finished', 'Finished Animating.')




if __name__ == '__main__':
    app = Window()
    app.mainloop()

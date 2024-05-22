import tkinter as tk
from tkinter import ttk

import numpy as np

# Implement the default Matplotlib key bindings.
import matplotlib
from matplotlib.backend_bases import key_press_handler, MouseButton
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from event_handler import EventHandler
from helper_misc import is_float, is_float_or_empty


class ViewWavelengthMapping:
    """
    Shows the mean value of each column in the rectangle selected in Picture
    (hc_view_image).
    """

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = fig = Figure()
        self._ax = ax = fig.add_subplot()

        ax.set_xlabel('x-coordinate (related to wavelength)')
        ax.set_ylabel('value')

        self._canvas = canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        self._toolbar = toolbar = NavigationToolbar2Tk(
            canvas, frame, pack_toolbar=False)
        toolbar._update_view = self._pyplot_home
        toolbar.update()

        canvas.mpl_connect("key_press_event", key_press_handler)

        canvas.get_tk_widget().grid(row=0, column=0, sticky='nesw')
        toolbar.grid(row=1, column=0, columnspan=2, sticky='ew')

        # set up side bar
        self.sbar = ttk.Frame(frame)
        self._populate_sbar()

        self.sbar.grid(
            row=0, column=1, sticky='nesw', padx=10, pady=10)

        EventHandler.bind('<<RectChanged>>', self.reset_limits)

    def reset_limits(self, *args):
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)

    def _pyplot_home(self, *args, **kwargs):
        self.reset_limits()
        self.update_plot()

    def add_calibration_point(self, var_nm, var_px):
        f = self._f_mapping
        idx = self._n_points

        validate = self._get_entry_validate_nm(var_nm)
        m_changed = self.c.get_fn_calibration_point_changed(idx)

        t = ttk.Entry(f, width=9, textvariable=var_nm,
                      validate='key', validatecommand=validate)
        t.grid(row=idx, column=0, padx=(10, 2))
        t.bind('<FocusIn>', self._entry_focusin)
        t.bind('<FocusOut>', m_changed)
        t.bind('<Return>', m_changed)

        ttk.Label(f, text='@').grid(row=idx, column=1)

        t = ttk.Entry(f, width=7, textvariable=var_px, validate='key',
                      validatecommand=self._validate_px)
        t.grid(row=idx, column=2, padx=(2, 10))
        t.bind('<FocusIn>', self._entry_focusin)
        t.bind('<FocusOut>', m_changed)
        t.bind('<Return>', m_changed)

        self._n_points += 1

    def _populate_sbar(self):
        sbar = self.sbar
        c = self.c

        ttk.Button(
            sbar, text='Switch Axes', command=self._switch_axes_fn
        ).grid(row=0, column=0, padx=10, pady=(10, 20))

        self._f_mapping = ttk.Frame(sbar)
        self._n_points = 0  # number of mapping points
        self._validate_px = (sbar.register(is_float_or_empty), '%P')

        if self.c.restrict is True:
            return

        ttk.Label(sbar, text='Wavelength Mapping').grid(
            row=1, column=0, padx=10, pady=(0, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=2, column=0, sticky='nesw', pady=(0, 5))

        self._f_mapping.grid(row=3, column=0)

        ttk.Button(
            sbar, text='Add', command=c.add_calibration_point
        ).grid(row=4, column=0, pady=5)

        ttk.Button(
            sbar, text='Update Response Functions',
            command=c.update_response_functions
        ).grid(row=5, column=0, pady=25, padx=10)

        ttk.Label(sbar, text='Export Data').grid(
            row=10, column=0, padx=10, pady=(10, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=11, column=0, sticky='nesw', pady=(0, 5))

        ttk.Button(
            sbar, text='Signal', command=c.export_signal
        ).grid(row=12, column=0, pady=5)

        ttk.Button(
            sbar, text='Signal of Files in Folder', command=c.export_signal_folder
        ).grid(row=13, column=0, pady=5)

    def _switch_axes_fn(self, *args):
        v = getattr(self, '_switch_axes', False)
        v = True if v is False else False
        self._switch_axes = v
        self.update_plot()

    def _entry_focusin(self, event):
        w = event.widget.select_range(0, 'end')

    def _get_entry_validate_nm(self, var):

        def fn(widget, newval, mode):
            w = self.f.nametowidget(widget)
            val = newval[:-2] if newval.endswith('nm') else newval

            if val == '':
                var.set('nm')
                w.icursor(0)
                return False

            if not is_float(val):
                return False

            if len(val.split('.')[0]) > 3:
                return False

            var.set(f'{val}nm')

            if mode == '1':  # insert
                w.icursor(w.index(tk.INSERT)+1)

            elif mode == '0': # delete
                w.icursor(w.index(tk.INSERT)-1)

            w.selection_clear()
            return False 

        fn_ = (self.f.register(fn), '%W', '%P', '%d')
        return fn_

    def update_plot(self, *args):
        if self.c.img is None:
            return

        vals, xvals = self.c.get_color_statistics()
        self._xvals_min, self._xvals_max = xvals[0], xvals[-1]
        nms = self.c.get_wavelengths(xvals=xvals)

        switch = nms is not None and getattr(self, '_switch_axes', False) is True
        x = nms if switch else xvals
        x_ = xvals if switch else nms

        ax = self._ax

        # save old limits so we can restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.cla()

        ax.plot(x, vals['mean_red'], color='red')
        ax.plot(x, vals['mean_green'], color='green')
        ax.plot(x, vals['mean_blue'], color='blue')

        # try to visualize variation in observed values
        v = 1.644  # 90% CI
        rl = vals['mean_red'] - v * vals['std_red']
        ru = vals['mean_red'] + v * vals['std_red']
        gl = vals['mean_green'] - v * vals['std_green']
        gu = vals['mean_green'] + v * vals['std_green']
        bl = vals['mean_blue'] - v * vals['std_blue']
        bu = vals['mean_blue'] + v * vals['std_blue']

        ax.plot(x, rl, color='red', alpha=0.75, linewidth=0.2)
        ax.plot(x, ru, color='red', alpha=0.75, linewidth=0.2)
        ax.plot(x, gl, color='green', alpha=0.75, linewidth=0.2)
        ax.plot(x, gu, color='green', alpha=0.75, linewidth=0.2)
        ax.plot(x, bl, color='blue', alpha=0.75, linewidth=0.2)
        ax.plot(x, bu, color='blue', alpha=0.75, linewidth=0.2)

        # plot wavelengths on secondary axis if available (or switch them)
        if nms is not None:
            def forward(v):
                return np.interp(v, x, x_)

            def inverse(v):
                return np.interp(v, x_, x)

            secax = ax.secondary_xaxis('top', functions=(forward, inverse))
            tw, tx = 'wavelength (nm)', 'x-coordinate (px)'
            secax.set_xlabel(tx if switch else tw)
            ax.set_xlabel(tw if switch else tx)

        else:
            ax.set_xlabel('x-coordinate (related to wavelength)')

        # plot valid calibration mappings
        if not self.c.restrict:
            wpm = self.c.get_valid_mappings()
            wpm = sorted(wpm)

            for nm, px in wpm:
                if switch is True:
                    px = inverse(px)

                ax.axvline(px, label=f'calibration line {nm}nm',
                           color='#999213', lw=1, alpha=0.9)

            if len(wpm) > 0:
                ax.legend()

        ax.set_ylabel('value')

        if xlim != (0, 1):
            _switch = getattr(self, '_switch', False) 

            if _switch is not switch:
                xlim = np.interp(xlim, x_, x)

            ax.set_xlim(*xlim)

        if ylim == (0, 1):
            ax.set_ylim(0, self.c.img.white_level)
        else:
            ax.set_ylim(*ylim)

        self._switch = switch
        self._canvas.draw()

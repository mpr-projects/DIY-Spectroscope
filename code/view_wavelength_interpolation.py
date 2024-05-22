import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler, MouseButton
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import numpy as np
import copy

from event_handler import EventHandler
import helper_interpolation


class ViewWavelengthInterpolation:
    """
    Plot the wavelength to pixel mapping so it's easier to see how
    non-linear the image is.
    """

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = fig = Figure()
        self._ax = ax = fig.add_subplot()

        ax.text(0.5, 0.5,
                'Mapping requires at least two points.',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

        ax.set_xlabel('x-coordinate (px)')
        ax.set_ylabel('wavelength (nm)')

        self._canvas = canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        self._toolbar = toolbar = NavigationToolbar2Tk(
            canvas, frame, pack_toolbar=False)
        toolbar.update()

        canvas.mpl_connect("key_press_event", key_press_handler)

        canvas.get_tk_widget().grid(row=0, column=0, sticky='nesw')
        toolbar.grid(row=1, column=0, columnspan=2, sticky='ew')

        # set up side bar
        self.sbar = ttk.Frame(frame)
        self.sbar.grid(
            row=0, column=1, sticky='nesw', padx=10, pady=10)
        self._populate_sbar()

        EventHandler.bind('<<RectChanged>>', self._reset_limits)

    def _populate_sbar(self):
        c, sbar = self.c, self.sbar

        ttk.Label(sbar, text='Interpolation Method').grid(
            row=0, column=0, padx=5, pady=(0, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=1, column=0, sticky='new')

        cb = ttk.Combobox(
            sbar, textvariable=c._var_interp_method, justify='center')
        cb.grid(row=2, column=0, padx=10, pady=10)
        cb.state(['readonly'])
        cb.bind('<<ComboboxSelected>>', self.c.handle_method_change)
        cb['values'] = list(helper_interpolation.mapping.keys())

    def _reset_limits(self, *args):
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)

    def update_plot(self, *args):
        ax = self._ax

        # save old limits so we can restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.cla()

        ax.set_xlabel('x-coordinate (px)')
        ax.set_ylabel('wavelength (nm)')

        m = self.c.get_valid_mappings()

        if len(m) < 2:
            ax.text(0.5, 0.5,
                    'Mapping requires at least two points.',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

            self._canvas.draw()
            return

        nms, pxs, cv = self.c.get_wavelengths(return_all=True)
        f = np.polynomial.polynomial.Polynomial.fit(cv[1], cv[0], 1)

        ax.plot(pxs, nms, label='interpolated wavelengths')
        ax.plot(pxs, f(pxs), linestyle='--', alpha=0.75, label='linear fit')
        ax.plot(cv[1], cv[0], 'o', label='calibration points', alpha=0.75)

        if self.c.img is not None:
            h, w = self.c.img.raw_image.shape[:2]
            ax.axvline(w//2, linestyle='--', color='black', lw=0.5, label='image center (horizontal)')

        ax.legend()

        if xlim != (0, 1):
            ax.set_xlim(*xlim)

        if ylim != (0, 1):
            ax.set_ylim(*ylim)

        self._canvas.draw()

import os
import copy
import numpy as np
from scipy.interpolate import CubicSpline

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler, MouseButton
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from event_handler import EventHandler
from helper_misc import is_float


class ViewSpectralResponse:
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

        ax.text(0.5, 0.5,
                'Spectral response curves can only be computed\nafter the'
                ' wavelengths have been mapped\nand an image has been loaded.',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')

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

    def _populate_sbar(self):
        c, sbar = self.c, self.sbar

        validate_float = (sbar.register(is_float), '%P')

        # load spectrum
        ttk.Label(sbar, text='Target Spectrum').grid(
            row=0, column=0, padx=10, pady=(0, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=1, column=0, sticky='new')

        b = ttk.Button(sbar, text='Load', command=c.load_spectrum)
        b.grid(row=2, column=0, pady=10)

        ttk.Label(sbar, text='or').grid(
            row=3, column=0, pady=1)

        f_ct = ttk.Frame(sbar)
        f_ct.grid(row=4, column=0, pady=(10, 0))

        ttk.Label(f_ct, text='Color Temperature').grid(
            row=0, column=0, padx=5, sticky='w')

        validate_ct = (sbar.register(self._validate_ct), '%P')

        e = ttk.Entry(
            f_ct, textvariable=c._var_color_temp, width=7,
            validate='key', validatecommand=validate_ct)
        e.grid(row=0, column=1, padx=5, sticky='w')
        e.bind('<FocusIn>', self._focusin_ct)
        e.bind('<Return>', c.load_color_temp)

        """
        ttk.Label(f_ct, text='UV Cut-Off').grid(
            row=1, column=0, padx=5, sticky='w')
        """

        validate_ctco = (sbar.register(self._validate_ctco), '%P')

        """
        e = ttk.Entry(
            f_ct, textvariable=c._var_color_temp_cutoff, width=7,
            validate='key', validatecommand=validate_ctco)
        e.grid(row=1, column=1, padx=5, sticky='w')
        e.bind('<FocusIn>', self._focusin_ct)
        e.bind('<Return>', c.load_color_temp)
        """

        b = ttk.Button(f_ct, text='Apply', command=c.load_color_temp)
        b.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Label(f_ct, text='Horizontal Offset').grid(
            row=3, column=0, sticky='w', pady=(10, 0))

        e = ttk.Entry(
            f_ct, textvariable=c._var_target_offset, width=7,
            validate='key', validatecommand=validate_ctco)
        e.grid(row=3, column=1, padx=5, sticky='w', pady=(10, 0))
        e.bind('<Return>', self.update_plot)

        ttk.Label(f_ct, text='Lower Cutoff').grid(row=4, column=0, sticky='w')

        e = ttk.Entry(
            f_ct, textvariable=c._var_target_cutoff_lower, width=7,
            validate='key', validatecommand=validate_ctco)
        e.grid(row=4, column=1, padx=5, sticky='w')
        e.bind('<Return>', self.update_plot)

        ttk.Label(f_ct, text='Upper Cutoff').grid(row=5, column=0, sticky='w')

        e = ttk.Entry(
            f_ct, textvariable=c._var_target_cutoff_upper, width=7,
            validate='key', validatecommand=validate_ctco)
        e.grid(row=5, column=1, padx=5, sticky='w')
        e.bind('<Return>', self.update_plot)

        # let user choose what's shown in the plot
        ttk.Label(sbar, text='Show in Plot').grid(
            row=5, column=0, pady=(25, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=6, column=0, sticky='new')

        f_sp = ttk.Frame(sbar)
        f_sp.grid(row=7, column=0)

        ttk.Checkbutton(
            f_sp, text='Sensor Red', variable=c._var_show_r,
            command=self.update_plot).grid(
                row=5, column=0, sticky='w', padx=10, pady=(10, 0))

        ttk.Checkbutton(
            f_sp, text='Sensor Green', variable=c._var_show_g,
            command=self.update_plot).grid(
                row=6, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            f_sp, text='Sensor Blue', variable=c._var_show_b,
            command=self.update_plot).grid(
                row=7, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            f_sp, text='Target Spectrum', variable=c._var_show_target,
            command=self.update_plot).grid(
                row=8, column=0, sticky='w', padx=10, pady=5)

        ttk.Checkbutton(
            f_sp, text='Response Red', variable=c._var_show_rr,
            command=self.update_plot).grid(
                row=9, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            f_sp, text='Response Green', variable=c._var_show_rg,
            command=self.update_plot).grid(
                row=10, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            f_sp, text='Response Blue', variable=c._var_show_rb,
            command=self.update_plot).grid(
                row=11, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            f_sp, text='Spectral Lines', variable=c._var_show_spectral_lines,
            command=self.update_plot).grid(
                row=12, column=0, sticky='w', padx=10, pady=5)

        # scale response functions
        ttk.Label(sbar, text='Scale Response Functions').grid(
            row=8, column=0, pady=(20, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=9, column=0, sticky='new')

        f_sr = ttk.Frame(sbar)
        f_sr.grid(row=10, column=0, pady=(9, 0))

        ttk.Label(f_sr, text='Scale Factor').grid(
            row=0, column=0, padx=5)

        e = ttk.Entry(
            f_sr, textvariable=c._var_scale, width=5,
            validate='key', validatecommand=validate_float)
        e.grid(row=0, column=1, padx=5)
        e.bind('<Return>', self.update_plot)

        f_sr = ttk.Frame(sbar)
        f_sr.grid(row=11, column=0, pady=(5, 0))

        ttk.Button(f_sr, text='Auto-Scale',
                   command=c.auto_scale_response
        ).grid(row=0, column=0, padx=5, pady=5)

        ttk.Button(f_sr, text='Apply', command=self.update_plot).grid(
            row=0, column=1, pady=5)

        # save the response functions
        ttk.Label(sbar, text='Save Response Functions').grid(
            row=13, column=0, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=14, column=0, sticky='new')

        f_rf = ttk.Frame(sbar)
        f_rf.grid(row=15, column=0, pady=(9, 0))

        ttk.Label(f_rf, text='Name').grid(
            row=0, column=0, sticky='w', padx=10)

        ttk.Entry(f_rf, textvariable=c._var_response_name).grid(
            row=0, column=1, sticky='w', padx=(0, 10))

        ttk.Button(f_rf, text='Save', command=c.save_response_fn).grid(
            row=2, column=0, columnspan=2, pady=10)

        # export the spectrum
        ttk.Label(sbar, text='Export Data').grid(
            row=20, column=0, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=21, column=0, sticky='new')

        ttk.Button(sbar, text='Target Spectrum', command=c.export_target).grid(
            row=22, column=0, pady=9)

    def _focusin_ct(self, event):
        n = len(self.c._var_color_temp.get())
        event.widget.icursor(n-1)

    def _validate_ct(self, newval):
        if newval == '':
            return  False

        if newval[-1] != 'K':
            return False

        if newval == 'K':
            return True

        val = newval[:-1]

        if len(val) > 4:
            return False

        return is_float(val)

    def _validate_ctco(self, newval):
        if newval == '':
            return False

        if newval[-2:] != 'nm':
            return False

        if newval == 'nm':
            return True

        val = newval[:-2]
        return is_float(val)

    def _plot_Fraunhofer(self, ax, wl, label=None):
        ax.axvline(wl, color='#5c1abf', linestyle='--', lw=0.5)

        if label is not None:
            ax.text(wl, 0.95, s=label, ha='center', backgroundcolor='white')

    def update_plot(self, *args):
        ax = self._ax

        # save old limits so we can restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.cla()

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')
        ax.set_ylim(0, 1)

        c = self.c
        average_vals = self.c.get_average_measured_values()

        if average_vals is None:
            ax.text(
                0.5, 0.5,
                'Spectral response curves can only be computed\nafter the'
                ' wavelengths have been mapped\nand an image has been loaded.',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

            self._canvas.draw()
            return

        nms, r, g, b = average_vals

        if c._var_show_r.get() is True:
            ax.plot(nms, r, color='red')

        if c._var_show_g.get() is True:
            ax.plot(nms, g, color='green')

        if c._var_show_b.get() is True:
            ax.plot(nms, b, color='blue')

        has_target = getattr(c, 'target_spectrum', None) is not None

        if has_target and c._var_show_target.get() is True:
            nms_ = c.target_spectrum[:, 0] - float(c._var_target_offset.get()[:-2])
            tgt_ = c.target_spectrum[:, 1]

            mask = np.logical_and(
                nms_ >= float(c._var_target_cutoff_lower.get()[:-2]),
                nms_ <= float(c._var_target_cutoff_upper.get()[:-2]))

            ax.plot(nms_[mask], tgt_[mask],
                    color='black', label='Target Spectrum')

            ax.legend()

        response_curves = c.get_response_curves()

        if response_curves is not None:
            rr, rg, rb = response_curves

            s = c.scale
            rr, rg, rb = s*rr, s*rg, s*rb

            if c._var_show_rr.get() is True:
                ax.plot(nms, rr, color='red', linestyle='--')

            if c._var_show_rg.get() is True:
                ax.plot(nms, rg, color='green', linestyle='--')

            if c._var_show_rb.get() is True:
                ax.plot(nms, rb, color='blue', linestyle='--')

        if c._var_show_spectral_lines.get() is True:
            # show some of the major spectral lines
            self._plot_Fraunhofer(ax, 430.79, 'G')
            self._plot_Fraunhofer(ax, 430.774, 'G')
            self._plot_Fraunhofer(ax, 438.355 , 'e')
            self._plot_Fraunhofer(ax, 486.134, 'F')
            self._plot_Fraunhofer(ax, 518.362, 'b1')
            self._plot_Fraunhofer(ax, 588.995)  # D2
            self._plot_Fraunhofer(ax, 589.592, 'D')  # D1
            self._plot_Fraunhofer(ax, 686.719, 'B')
            self._plot_Fraunhofer(ax, 656.281, 'C')
            self._plot_Fraunhofer(ax, 527.039 , 'E2')

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')

        if xlim != (0, 1):
            ax.set_xlim(*xlim)

        if ylim != (0, 1):
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, 1.1)

        self._canvas.draw()

import copy
import numpy as np
from scipy.interpolate import CubicSpline
import warnings

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler, MouseButton
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from event_handler import EventHandler
import helper_color_temp
from helper_misc import is_float



class ViewSpectrum:
    """
    Shows the spectrum.
    """

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = fig = Figure()
        self._ax = ax = fig.add_subplot()

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')

        ax.text(0.5, 0.5,
                'Please finish the calibration process and load an image.',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

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

        # let user choose what's shown in the plot
        ttk.Label(sbar, text='Show in Plot').grid(
            row=0, column=0, pady=(9, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=1, column=0, sticky='new')

        f_sp = ttk.Frame(sbar)
        f_sp.grid(row=2, column=0)

        ttk.Checkbutton(
            f_sp, text='Combined Spectrum', variable=c._var_show_spectrum,
            command=self.update_plot).grid(
                row=4, column=0, sticky='w', padx=9, pady=(9, 0))

        ttk.Checkbutton(
            f_sp, text='Spectrum Components', variable=c._var_show_components,
            command=self.update_plot).grid(
                row=5, column=0, sticky='w', padx=9)

        ttk.Checkbutton(
            f_sp, text='Sensor', variable=c._var_show_sensor,
            command=self.update_plot).grid(
                row=6, column=0, sticky='w', padx=9)

        ttk.Checkbutton(
            f_sp, text='Response Functions', variable=c._var_show_response,
            command=self.update_plot).grid(
                row=7, column=0, sticky='w', padx=9)

        ttk.Checkbutton(
            f_sp, text='Reference Spectrum', variable=c._var_show_reference,
            command=self.update_plot).grid(
                row=8, column=0, sticky='w', padx=9)

        # scalings
        ttk.Label(sbar, text='Sensor Scale').grid(
            row=9, column=0, padx=5, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=10, column=0, sticky='new')

        f = ttk.Frame(sbar)
        f.grid(row=11, column=0)

        dc = ttk.Scale(
            f, orient='horizontal', variable=c._var_sensor_scale,
            length=100, command=self.update_plot, from_=0.5, to=1.5)
        dc.grid(row=0, column=0, padx=5, pady=10)

        validate_float = (sbar.register(is_float), '%P')

        e = ttk.Entry(
            f, textvariable=c._var_sensor_scale, width=5,
            validate='key', validatecommand=validate_float)
        e.grid(row=0, column=1, padx=5)
        e.bind('<Return>', self.update_plot)

        ttk.Label(sbar, text='Spectrum Scale').grid(
            row=12, column=0, padx=5, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=13, column=0, sticky='new')

        f = ttk.Frame(sbar)
        f.grid(row=14, column=0)

        dc = ttk.Scale(
            f, orient='horizontal', variable=c._var_spectrum_scale,
            length=100, command=self.update_plot, from_=0.5, to=1.5)
        dc.grid(row=0, column=0, padx=5, pady=10)

        e = ttk.Entry(
            f, textvariable=c._var_spectrum_scale, width=5,
            validate='key', validatecommand=validate_float)
        e.grid(row=0, column=1, padx=5)
        e.bind('<Return>', self.update_plot)

        # load spectrum
        ttk.Label(sbar, text='Reference Spectrum').grid(
            row=15, column=0, padx=5, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=16, column=0, sticky='new')

        b = ttk.Button(sbar, text='Load', command=c.load_spectrum)
        b.grid(row=17, column=0, pady=10)

        ttk.Label(sbar, text='or').grid(
            row=18, column=0, pady=1)

        f_ct = ttk.Frame(sbar)
        f_ct.grid(row=19, column=0, pady=(10, 0))

        ttk.Label(f_ct, text='Color Temperature').grid(
            row=0, column=0, padx=5)

        validate_ct = (sbar.register(self._validate_ct), '%P')

        e = ttk.Entry(
            f_ct, textvariable=c._var_color_temp, width=5,
            validate='key', validatecommand=validate_ct)
        e.grid(row=0, column=1, padx=5)
        e.bind('<FocusIn>', self._focusin_ct)
        e.bind('<Return>', c.load_color_temp)

        b = ttk.Button(f_ct, text='Apply', command=c.load_color_temp)
        b.grid(row=1, column=0, columnspan=2, pady=5)

        # save the spectrum
        ttk.Label(sbar, text='Save Spectrum').grid(
            row=23, column=0, pady=(25, 6))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=24, column=0, sticky='new')

        f_rf = ttk.Frame(sbar)
        f_rf.grid(row=25, column=0, pady=(9, 0))

        ttk.Label(f_rf, text='Name').grid(
            row=0, column=0, sticky='w', padx=10)

        ttk.Entry(f_rf, textvariable=c._var_spectrum_name).grid(
            row=0, column=1, sticky='w', padx=(0, 10))

        ttk.Button(f_rf, text='Save', command=c.save_spectrum).grid(
            row=2, column=0, columnspan=2, pady=10)

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

    def _plot_component(self, ax, nms, vals, scale, color_idx):
        crf = self.c.calibration.combined_response_function

        if crf is None:
            return

        weights = crf.get_weights()
        weights = weights[color_idx]

        nms = np.array(nms)
        vals = np.array(vals)

        mask = (weights != 0)
        nms = nms[mask]
        vals = vals[mask]
        c = ['red', 'green', 'blue'][color_idx]

        ax.plot(nms, vals*scale, color=c, alpha=0.5)

    def update_plot(self, *args):
        ax = self._ax
        ax.cla()

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')
        ax.set_ylim(0, 1.1)

        crf = self.c.calibration.combined_response_function

        if crf is None or self.c.img is None:
            ax.text(0.5, 0.5,
                    'Please finish the calibration process and load an image.',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

            self._canvas.draw()
            return

        # the nms of the signal can change, depending on how the spectroscope
        # is currently mounted; that of the response functions is fixed once
        # calibration is finished; most plotting happens using the nms of the
        # combined response function, only the signal gets plotted with its
        # specific nms

        c = self.c
        nms_s, r, g, b = c.get_average_measured_values()
        scale = c._var_sensor_scale.get() 

        if c._var_show_sensor.get() is True:
            ax.plot(nms_s, r*scale, color='red')
            ax.plot(nms_s, g*scale, color='green')
            ax.plot(nms_s, b*scale, color='blue')

        nms, rr, rg, rb = crf.nms, crf.r, crf.g, crf.b

        if c._var_show_response.get() is True:
            ax.plot(nms, rr, color='red', linestyle='--')
            ax.plot(nms, rg, color='green', linestyle='--')
            ax.plot(nms, rb, color='blue', linestyle='--')

        components, combined = c.compute_spectrum(
            sensor_rgb=[nms_s, r, g, b], response_rgb=[nms, rr, rg, rb])
        scale = c._var_spectrum_scale.get() 

        if c._var_show_spectrum.get() is True:
            ax.plot(nms, combined*scale, color='black')

        if c._var_show_components.get() is True:
            for idx, vals in enumerate(components):
                self._plot_component(ax, nms, vals, scale, idx)
            """
            for c, col in zip(components, ['red', 'green', 'blue']):
                ax.plot(nms, c, color=col, alpha=0.5)
            """

        has_reference = getattr(c, 'reference_spectrum', None) is not None

        if has_reference and c._var_show_reference.get() is True:
            ref_nms = c.reference_spectrum[:, 0]
            ref_vals = c.reference_spectrum[:, 1]
            ref_vals /= ref_vals[c.spectrum_max_idx]
            ax.plot(ref_nms, ref_vals, color='black', linestyle='--', alpha=0.5)

        # I don't want to change zoom and pan; when ax is updated then the
        # navigation stack of the toolbar (self._toolbar._nav_stack) doesn't
        # change but the view is not automatically updated to the last values;
        # by moving forward in the navigation stack (we're already at the
        # last position -- self._toolbar._nav_stack._pos) seems to update the
        # view
        self._toolbar.forward()

        self._canvas.draw()

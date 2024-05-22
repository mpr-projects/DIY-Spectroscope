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
from helper_misc import is_float_or_empty


class ViewResponseMixing:

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = fig = Figure()
        self._ax = ax = fig.add_subplot()

        ax.text(0.5, 0.5,
                'No Response Functions have been added.',
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
        vf = (sbar.register(is_float_or_empty), '%P')

        ttk.Label(sbar, text='Channel Mixing').grid(
            row=0, column=0, padx=10, pady=(0, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=1, column=0, sticky='nesw')

        f = ttk.Frame(sbar)
        f.grid(row=2, column=0)

        # blue 
        ttk.Label(f, text='Blue').grid(
            row=0, column=0, sticky='w', padx=10, pady=(10, 5))

        # blue fade in
        ttk.Label(f, text='Fade In').grid(
            row=1, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_in_start_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=1, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=1, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_in_end_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=1, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # blue fade out
        ttk.Label(f, text='Fade Out').grid(
            row=2, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_out_start_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=2, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=2, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_out_end_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=2, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # green
        ttk.Label(f, text='Green').grid(
            row=3, column=0, columnspan=5, sticky='w', padx=5, pady=(5, 0))

        # green fade in
        ttk.Label(f, text='Fade In').grid(
            row=4, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_in_start_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=4, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=4, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_in_end_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=4, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # green fade out
        ttk.Label(f, text='Fade Out').grid(
            row=5, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_out_start_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=5, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=5, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_out_end_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=5, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # red
        ttk.Label(f, text='Red').grid(
            row=6, column=0, columnspan=5, sticky='w', padx=5, pady=(5,0))

        # red fade in
        ttk.Label(f, text='Fade In').grid(
            row=7, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_in_start_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=7, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=7, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_in_end_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=7, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # red fade out
        ttk.Label(f, text='Fade Out').grid(
            row=8, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_fade_out_start_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=8, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        ttk.Label(f, text='-').grid(row=8, column=2)

        e = ttk.Entry(f, textvariable=c._var_fade_out_end_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=8, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_channel_mixing_change)
        e.bind('<FocusOut>', c.handle_channel_mixing_change)

        # plotting checkboxes
        f = ttk.Frame(sbar)
        f.grid(row=3, column=0, pady=(10, 0))

        cb = ttk.Checkbutton(
            f, text='plot combined functions', variable=c._var_plot_combined,
            command=self.update_plot)
        cb.grid(row=0, column=0, sticky='w')

        cb = ttk.Checkbutton(
            f, text='plot fading boundaries', variable=c._var_plot_boundaries,
            command=self.update_plot)
        cb.grid(row=1, column=0, sticky='w')

        # function details
        ttk.Label(sbar, text='Function Details').grid(
            row=4, column=0, padx=10, pady=(25, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=5, column=0, sticky='nesw')

        self._cb = cb = ttk.Combobox(
            sbar, textvariable=c._var_d_name, justify='center')
        cb.grid(row=6, column=0, padx=10, pady=(10, 0))
        cb.state(['readonly'])

        f = ttk.Frame(sbar)
        f.grid(row=7, column=0, pady=(10, 0))

        ttk.Checkbutton(
            f, text='plot', variable=c._var_d_plot,
            command=c.handle_function_details_change_plot
        ).grid(row=0, column=0, padx=5)
                             
        ttk.Checkbutton(
            f, text='use', variable=c._var_d_use,
            command=c.handle_function_details_change_use
        ).grid(row=0, column=1, padx=5)

        ttk.Label(f, text='Scale').grid(row=0, column=2, padx=(3, 0))

        e = ttk.Entry(f, textvariable=c._var_d_scale, width=4,
                      validate='key', validatecommand=vf)
        e.grid(row=0, column=3, padx=5)
        e.bind('<Return>',c.handle_function_details_change_scale)
        e.bind('<FocusOut>', c.handle_function_details_change_scale)

        f = ttk.Frame(sbar)
        f.grid(row=8, column=0)

        # blue 
        ttk.Label(f, text='Blue').grid(
            row=0, column=0, sticky='w', padx=10, pady=(10, 5))

        # blue fade in
        ttk.Label(f, text='Fade In').grid(
            row=1, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_start_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=1, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=1, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_end_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=1, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # blue fade out
        ttk.Label(f, text='Fade Out').grid(
            row=2, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_start_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=2, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=2, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_end_b, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=2, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # green
        ttk.Label(f, text='Green').grid(
            row=3, column=0, columnspan=5, sticky='w', padx=5, pady=(5, 0))

        # green fade in
        ttk.Label(f, text='Fade In').grid(
            row=4, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_start_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=4, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=4, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_end_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=4, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # green fade out
        ttk.Label(f, text='Fade Out').grid(
            row=5, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_start_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=5, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=5, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_end_g, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=5, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # red
        ttk.Label(f, text='Red').grid(
            row=6, column=0, columnspan=5, sticky='w', padx=5, pady=(5,0))

        # red fade in
        ttk.Label(f, text='Fade In').grid(
            row=7, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_start_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=7, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=7, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_in_end_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=7, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # red fade out
        ttk.Label(f, text='Fade Out').grid(
            row=8, column=0, sticky='w', padx=(20, 5))

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_start_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=8, column=1, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        ttk.Label(f, text='-').grid(row=8, column=2)

        e = ttk.Entry(f, textvariable=c._var_d_fade_out_end_r, width=6,
                      validate='key', validatecommand=vf)
        e.grid(row=8, column=3, padx=(5, 10))
        e.bind('<Return>', c.handle_function_details_change)
        e.bind('<FocusOut>', c.handle_function_details_change)

        # delete response function
        ttk.Button(
            sbar, text='Delete', command=c.delete_response_function
        ).grid(row=9, column=0, padx=10, pady=10)

        # allow importing response functions
        ttk.Button(
            sbar, text='Import', command=c.import_response_function
        ).grid(row=10, column=0, padx=10, pady=0)

        # export data
        ttk.Label(sbar, text='Export Data').grid(
            row=20, column=0, padx=10, pady=(25, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=21, column=0, sticky='nesw')

        f = ttk.Frame(sbar)
        f.grid(row=22, column=0, pady=(10, 0))
        
        ttk.Button(
            f, text='Combined Response Function', command=c.export_crf
        ).grid(row=0, column=0, pady=5)

        ttk.Button(
            f, text='Component Weights', command=c.export_weights
        ).grid(row=1, column=0, pady=5)

    def set_response_functions_options(self, vals):
        self._cb['values'] = vals

    def update_plot(self):
        ax = self._ax

        # save old limits so we can restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.cla()

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('value (normalized)')

        if len(self.c.calibration.response_functions) == 0:
            ax.text(0.5, 0.5,
                    'No Response Functions have been added.',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

            self._canvas.draw()
            return

        self._plot_response_functions()

        if self.c._var_plot_boundaries.get() is True:
            self._plot_fade_boundaries()

        if self.c._var_plot_combined.get() is True:
            self._plot_combined_response_function()

        if xlim != (0, 1):
            ax.set_xlim(*xlim)

        if ylim != (0, 1):
            ax.set_ylim(*ylim)

        self._canvas.draw()

    def _plot_combined_response_function(self):
        rf = self.c.calibration.combined_response_function

        if rf is None:
            return

        ax = self._ax
        ax.plot(rf.nms, rf.r, color='r', alpha=0.25)
        ax.plot(rf.nms, rf.g, color='g', alpha=0.25)
        ax.plot(rf.nms, rf.b, color='b', alpha=0.25)

    def _plot_response_functions(self):
        color = dict(r='red', g='green', b='blue')

        for rf in self.c.calibration.response_functions:
            if rf.plot is False:
                continue

            nms, scale = rf.nms, rf.scale
            weights = rf.get_weights()

            for c, w in zip(['r', 'g', 'b'], weights):
                v = scale * getattr(rf, c)
                main = np.where(w == 1, v, float('nan'))
                self._ax.plot(nms, main, color=color[c])

                mask = np.logical_and(w > 0, w < 1)
                fade = np.where(mask, v, float('nan'))
                self._ax.plot(
                    nms, fade, color=color[c], linestyle='--', lw=0.5)

    def _plot_fade_boundaries(self):
        for c in ['r', 'g', 'b']:
            for io in ['in', 'out']:
                self._plot_fade_boundaries_impl(c, io)

    def _plot_fade_boundaries_impl(self, c, io):
        ax = self._ax
        color = dict(r='red', g='green', b='blue')[c]
        start = getattr(self.c, f'_var_fade_{io}_start_{c}').get()
        end = getattr(self.c, f'_var_fade_{io}_end_{c}').get()

        start = '' if start == '' else float(start)
        end = '' if end == '' else float(end)

        if start == '' and end == '':
            return

        if start != '' and end != '' and start > end:
            return

        if start == end:
            end = ''

        if start != '':
            ax.axvline(start, linestyle='--', color=color, alpha=0.3)

        if end != '':
            ax.axvline(end, linestyle='--', color=color, alpha=0.3)

        if start != '' and end != '':
            ax.axvspan(start, end, color=color, alpha=0.1)

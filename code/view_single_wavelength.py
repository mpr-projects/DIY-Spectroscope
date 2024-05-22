import tkinter as tk
from tkinter import ttk
import numpy as np

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from event_handler import EventHandler


class ViewSingleWavelength:
    """
    Shows the distribution of colors at a single column in the photograph. When
    calibrated then the column should correspond to a single wavelength. The
    dashed line in the Picture (in hc_view_image) indicates which wavelength is
    used. The top and bottom lines of the rectangle in the Picture determine
    the height of the column to be used.
    """

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = fig = Figure()
        self._ax = ax = fig.add_subplot()

        ax.set_xlabel('intensity')
        ax.set_ylabel('# observations')

        self._canvas = canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        self._toolbar = toolbar

        canvas.mpl_connect("key_press_event", key_press_handler)

        canvas.get_tk_widget().grid(row=0, column=0, sticky='nesw')
        toolbar.grid(row=1, column=0, columnspan=2, sticky='ew')

        # sidebar
        sb_frame = ttk.Frame(frame)
        sb_frame.grid(row=0, column=1, sticky='nesw', padx=10, pady=10)

        self._build_gui(sb_frame)

        EventHandler.bind('<<RectChanged>>', self._reset_limits)

    def _build_gui(self, f):
        c = self.c

        ttk.Label(f, text='Choose Plot').grid(
            row=0, column=0, padx=10, pady=(0, 5))

        ttk.Separator(f, orient='horizontal').grid(
            row=1, column=0, sticky='nesw')

        f_ = ttk.Frame(f)
        f_.grid(row=2, column=0)

        v = c._var_plot
        v.set('lineplot')

        def change_plot(*args):
            self._reset_limits()
            self.update_plot()

        ttk.Radiobutton(
            f_, text='Histogram', variable=v, value='histogram',
            command=change_plot
        ).grid(row=0, column=0, sticky='w', padx=20, pady=(5, 0))

        ttk.Radiobutton(
            f_, text='Line Plot', variable=v, value='lineplot',
            command=change_plot
        ).grid(row=1, column=0, sticky='w', padx=20)

        ttk.Button(f, text='Reset Limits', command=change_plot).grid(
            row=3, column=0, pady=(7, 0))

        ttk.Label(f, text='Set Dashed Coordinate').grid(
            row=4, column=0, padx=10, pady=(20, 5))

        ttk.Separator(f, orient='horizontal').grid(
            row=5, column=0, sticky='nesw')

        self._sc_dc = ttk.Scale(
            f, orient='horizontal', variable=c._var_dashed,
            length=150, command=c.handle_dashed_coord_change)

        self._sc_dc.grid(row=6, column=0, pady=10, padx=10)

        ttk.Button(
            f, text='Save Line Plot Data',
            command=c.save_lineplot_data
        ).grid(row=7, column=0, padx=10, pady=10)

    def _reset_limits(self, *args):
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)

    def configure_dashed_scale(self, from_, to):
        self._sc_dc.configure(from_=from_, to=to)

    def _plot_histogram(self, ax, vals):
        _, _, raw_image, raw_colors = vals
        vals = list()

        for i in range(3):
            vals.append(raw_image[raw_colors == i])

        ax.hist(vals[0], bins=30, histtype='step', color='red')
        ax.hist(vals[1], bins=30, histtype='step', color='green')
        ax.hist(vals[2], bins=30, histtype='step', color='blue')

        ax.axvline(np.mean(vals[0]), color='red', alpha=0.75, ls='--', linewidth=0.2)
        ax.axvline(np.mean(vals[1]), color='green', alpha=0.75, ls='--', linewidth=0.2)
        ax.axvline(np.mean(vals[2]), color='blue', alpha=0.75, ls='--', linewidth=0.2)

        ax.set_xlabel('intensity')
        ax.set_ylabel('# observations')

        ax.set_xlim(0, self.c.img.white_level)

    def _plot_lineplot(self, ax, vals):
        t, b, raw_image, raw_colors = vals

        pxs = np.arange(t, b)
        r = np.where(raw_colors == 0, raw_image, float('nan'))
        g = np.where(raw_colors == 1, raw_image, float('nan'))
        b = np.where(raw_colors == 2, raw_image, float('nan'))

        ax.plot(pxs, r, 'o', ms=0.5, color='red')
        ax.plot(pxs, g, 'o', ms=0.5, color='green')
        ax.plot(pxs, b, 'o', ms=0.5, color='blue')

        ax.set_xlabel('y-coordinate')
        ax.set_ylabel('value')
        # ax.set_xlim(pxs[0], pxs[1])  # doesn't always update automatically due to forward

    def update_plot(self, *args):
        if self.c.img is None:
            return

        vals = self.c.get_dashed_data()
        ax = self._ax

        # save old limits so we can restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.cla()

        if self.c.plot_type == 'histogram':
            self._plot_histogram(ax, vals)

        else:
            self._plot_lineplot(ax, vals)

            if ylim == (0, 1):
                ylim = (0, self.c.img.white_level)

        if xlim != (0, 1):
            ax.set_xlim(*xlim)

        if ylim != (0, 1):
            ax.set_ylim(*ylim)

        dc = self.c.dashed_coord
        title = f'X-Coordinate {dc}'

        nm = self.c.get_wavelengths(xvals=np.array([dc]))

        if nm is not None:
            title += f' ({nm[0]:.2f}nm)'

        ax.set_title(title)
        self._canvas.draw()

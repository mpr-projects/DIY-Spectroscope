import tkinter as tk
import numpy as np

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_single_wavelength import ViewSingleWavelength


class ControllerSingleWavelength(ControllerHelper):

    def __init__(self, controller, frame):
        super().__init__(controller, frame)

        self._set_up_variables()
        self._view = ViewSingleWavelength(self, frame)
        self._set_up_bindings()

    @property
    def plot_type(self):
        return self._var_plot.get()

    @property
    def dashed_coord(self):
        return self._var_dashed.get()

    def _set_up_variables(self):
        self._var_plot = tk.StringVar()
        self._var_dashed = tk.IntVar()

    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)
        EventHandler.bind(
            '<<ControllerLoadedState>>', self._loaded_state_callback)

    def _reload_variables(self):
        t, b, l, r = self.calibration.rectangle_coords
        dc = self.calibration.dashed_coord

        if dc is not None:
            self._var_dashed.set(dc)
            self._view.configure_dashed_scale(
                from_=l, to=r)

    def _loaded_state_callback(self, *args):
        self._reload_variables()

    def _clicked_tab(self, params):
        if params.clicked_name != 'Single Wavelength':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._reload_variables()
        self._view.update_plot()

    def handle_image_change(self):
        self._view.update_plot()

    def handle_dashed_coord_change(self, *args):
        c = self._var_dashed.get()
        c = int(round(c))
        self._var_dashed.set(c)

        if c == self.calibration.dashed_coord:
            return

        self._c.has_changed = True
        self.calibration.dashed_coord = c

        self._view.update_plot()

    def save_lineplot_data(self):
        if self.img is None:
            return

        fname = tk.filedialog.asksaveasfilename()

        if len(fname) == 0:
            return

        t, b, raw_image, raw_colors = self.get_dashed_data()

        pxs = np.arange(t, b)
        r = np.where(raw_colors == 0, raw_image, float('nan'))
        g = np.where(raw_colors == 1, raw_image, float('nan'))
        b = np.where(raw_colors == 2, raw_image, float('nan'))

        c = np.stack((pxs, r, g, b), axis=0)
        np.savetxt(fname, c)

    def get_dashed_data(self):
        t, b, l, r = self.calibration.rectangle_coords
        dc = self.calibration.dashed_coord

        raw_image = self.img.raw_image[t:b, dc]
        raw_colors = self.img.raw_colors[t:b, dc]

        return t, b, raw_image, raw_colors


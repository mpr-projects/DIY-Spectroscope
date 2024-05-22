import tkinter as tk
from tkinter import messagebox

import os
import numpy as np
from scipy.interpolate import CubicSpline

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_spectral_response import ViewSpectralResponse
import helper_color_temp



class ControllerSpectralResponse(ControllerHelper):

    def __init__(self, controller, frame):
        super().__init__(controller, frame)

        self._set_up_variables()
        self._view = ViewSpectralResponse(self, frame)
        self._set_up_bindings()

    def _set_up_variables(self):
        self._var_color_temp = tk.StringVar(value='K')
        self._var_color_temp_cutoff = tk.StringVar(value='nm')
        self._var_target_offset = tk.StringVar(value='0nm')
        self._var_target_cutoff_lower = tk.StringVar(value='400nm')
        self._var_target_cutoff_upper = tk.StringVar(value='700nm')

        self._var_show_r = tk.BooleanVar(value=True)
        self._var_show_g = tk.BooleanVar(value=True)
        self._var_show_b = tk.BooleanVar(value=True)
        self._var_show_target = tk.BooleanVar(value=True)
        self._var_show_rr = tk.BooleanVar(value=False)
        self._var_show_rg = tk.BooleanVar(value=False)
        self._var_show_rb = tk.BooleanVar(value=False)
        self._var_show_spectral_lines = tk.BooleanVar(value=False)

        self._var_scale = tk.DoubleVar()
        self._var_scale.set(1)

        self._var_response_name = tk.StringVar()

    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)
        EventHandler.bind(
            '<<ControllerLoadedState>>', self._view.update_plot)

    @property
    def scale(self):
        return self._var_scale.get()

    def _clicked_tab(self, params):
        if params.clicked_name != 'Spectral Response':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._view.update_plot()

    def handle_image_change(self):
        fname = os.path.basename(self.img.fpath)
        self._var_response_name.set(fname)
        self._view.update_plot()

    def load_spectrum(self):
        fname = tk.filedialog.askopenfilename()

        if len(fname) == 0:
            return

        data = np.loadtxt(fname)

        if not data.ndim == 2 or data.shape[1] != 2:
            messagebox.showerror(title='Error', message=(
                'The spectrum must have exactly two columns.'))
            return

        data[:, 1] /= data[:, 1].max()
        self.target_spectrum = data
        self._view.update_plot()

    def load_color_temp(self, *args):
        if len(self._c.get_valid_mappings()) < 2:
            return

        T = self._var_color_temp.get()

        if T == 'K':
            return

        T = float(T[:-1])
        nms = self._c.get_wavelengths()
        val = helper_color_temp.get_radiance(T, nms)
        cutoff = self._var_color_temp_cutoff.get()
        cutoff = 0 if cutoff == 'nm' else float(cutoff[:-2])
        val[nms < cutoff] = 0
        self.target_spectrum = np.stack((nms, val), axis=-1)
        self._view.update_plot()

    def get_response_curves(self):
        if getattr(self, 'target_spectrum', None) is None:
            return None

        nms, r, g, b = self.get_average_measured_values()
        target = self.target_spectrum
        target_offset = float(self._var_target_offset.get()[:-2])

        # Todo: maybe replace with linear interpolation (for uv cutoff)
        cs = CubicSpline(target[:, 0] - target_offset, target[:, 1])
        target_vals = cs(nms)

        mask = np.isclose(target_vals, 0)
        target_vals[mask] = 1

        rr = r / target_vals
        rg = g / target_vals
        rb = b / target_vals

        rr[mask] = 0
        rg[mask] = 0
        rb[mask] = 0

        max_val = max([rr.max(), rg.max(), rb.max()])
        rr /= max_val
        rg /= max_val
        rb /= max_val

        cutoff_lower = float(self._var_target_cutoff_lower.get()[:-2])
        cutoff_upper = float(self._var_target_cutoff_upper.get()[:-2])
        mask = np.logical_or(nms < cutoff_lower, nms > cutoff_upper)

        rr[mask] = 0
        rg[mask] = 0
        rb[mask] = 0

        # scale response functions
        max_val = max([np.amax(c) for c in [rr, rg, rb]])
        self._var_scale.set(1 / max_val)

        return rr, rg, rb

    def auto_scale_response(self):
        # get the response functions, set the maximum to 1
        rf = self.get_response_curves()

        if rf is None:
            return

        max_val = max([np.amax(c) for c in rf])
        self._var_scale.set(1 / max_val)

    def save_response_fn(self):
        name = self._var_response_name.get()

        if name == '':
            tk.messagebox.showerror(title='Error', message=(
                'Please enter a name.'))
            return

        nms = self.get_wavelengths(return_all=True)
        rf = self.get_response_curves()

        if rf is None or nms is None:
            tk.messagebox.showerror(title='Error', message=(
                'You first need to finish the pixel mapping'
                ' and load a spectrum.'))
            return

        nms, xvals, _ = nms
        r, g, b = rf

        s = self._var_scale.get()
        r, g, b = s*r, s*g, s*b

        self.calibration.add_response_function(name, xvals, nms, r, g, b)

    def export_target(self):
        has_target = getattr(self, 'target_spectrum', None) is not None

        if has_target is False:
            return

        ftypes = [('Binary', '*.npy'), ('Text File', '*.txt')]
        fname = tk.filedialog.asksaveasfilename(filetypes=ftypes)

        if len(fname) == 0:
            return

        nms = self.target_spectrum[:, 0] - float(self._var_target_offset.get()[:-2])
        tgt = self.target_spectrum[:, 1]

        # Todo: should interpolate target spectrum at same nms as used in 
        #       response functions (so the data is consistent for the ml model)

        mask = np.logical_or(
            nms < float(self._var_target_cutoff_lower.get()[:-2]),
            nms > float(self._var_target_cutoff_upper.get()[:-2]))

        tgt[mask] = 0
        tgt = np.stack((nms, tgt), axis=0)
        save_fn = np.savetxt if fname[-3:] == 'txt' else np.save
        save_fn(fname, tgt)

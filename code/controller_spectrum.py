import tkinter as tk
from tkinter import messagebox

import os
import numpy as np

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_spectrum import ViewSpectrum
from helper_interpolation import cubic_spline, pw_linear
import helper_color_temp



class ControllerSpectrum(ControllerHelper):

    def __init__(self, controller, frame):
        super().__init__(controller, frame)

        self._set_up_variables()
        self._view = ViewSpectrum(self, frame)
        self._set_up_bindings()

    def _set_up_variables(self):
        self._var_show_spectrum = tk.BooleanVar(value=False)
        self._var_show_components = tk.BooleanVar(value=True)
        self._var_show_sensor = tk.BooleanVar(value=False)
        self._var_show_response = tk.BooleanVar(value=False)
        self._var_show_reference = tk.BooleanVar(value=False)

        self._var_sensor_scale = tk.DoubleVar(value=1)
        self._var_spectrum_scale = tk.DoubleVar(value=1)

        self._var_color_temp = tk.StringVar(value='K')
        self._var_spectrum_name = tk.StringVar()


    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)

        EventHandler.bind(
            '<<ControllerLoadedState>>', self._loaded_state_callback)

    def _clicked_tab(self, params):
        if params.clicked_name != 'Spectrum':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._view.update_plot()

    def _loaded_state_callback(self, *args):
        self._view.update_plot()

    def handle_image_change(self):
        fname = os.path.basename(self.img.fpath)
        fname = '.'.join(fname.split('.')[:-1])
        self._var_spectrum_name.set(fname)
        self._view.update_plot()

    def load_spectrum(self):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        fname = tk.filedialog.askopenfilename()

        if len(fname) == 0:
            return

        data = np.loadtxt(fname)

        if not data.ndim == 2 or data.shape[1] != 2:
            messagebox.showerror(title='Error', message=(
                'The spectrum must have exactly two columns.'))
            return

        nms = crf.nms

        data = cubic_spline(data[:, 0], data[:, 1], nms)
        self.reference_spectrum = np.stack((nms, data), axis=-1)

        self._view.update_plot()

    def load_color_temp(self, *args):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        T = self._var_color_temp.get()
        T = float(T[:-1])
        nms = crf.nms
        val = helper_color_temp.get_radiance(T, nms)
        self.reference_spectrum = np.stack((nms, val), axis=-1)
        self._view.update_plot()

    def save_spectrum(self):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        name = self._var_spectrum_name.get()

        file = tk.filedialog.asksaveasfilename(
            filetypes=[('CSV', '*.csv')], initialfile=name)

        if len(file) == 0:
            return

        _, combined = self.compute_spectrum() 
        data = np.stack((crf.nms, combined))
        np.savetxt(file, data, delimiter=',')

    def compute_spectrum(self, sensor_rgb=None, response_rgb=None):
        """
        Computes the spectrum. The arguments should both contain four
        components: [nms, r, g, b]. The returned spectrum uses the nms of the
        combined response function.
        """
        if sensor_rgb is None:
            sensor_rgb = self.get_average_measured_values()

        if response_rgb is None:
            crf = self.calibration.combined_response_function
            response_rgb = [crf.nms, crf.r, crf.g, crf.b]

        nms_s, sensor_rgb = sensor_rgb[0], sensor_rgb[1:]
        nms_r, response_rgb = response_rgb[0], response_rgb[1:]

        if (nms_s != nms_r).any():
            # map sensor data to nms used in response function
            # sensor_rgb = pw_linear(nms_r, response_rgb, nms_s, y_axis=1)
            sensor_rgb = pw_linear(nms_s, sensor_rgb, nms_r, y_axis=1)

        masks = [np.logical_or(np.isclose(s, 0), np.isclose(r, 0))
                 for s, r in zip(sensor_rgb, response_rgb)]

        sensor_rgb = [np.where(m, 0, s) for m, s in zip(masks, sensor_rgb)]
        response_rgb = [np.where(m, 1, r) for m, r in zip(masks, response_rgb)]
        components = [s/r for s, r in zip(sensor_rgb, response_rgb)]

        crf = self.calibration.combined_response_function
        weights = crf.get_weights()

        combined = sum([w*v for w, v in zip(weights, components)])
        s = sum(weights)
        s = np.where(s == 0, 1, s)
        combined /= s

        self.spectrum_max_idx = np.argmax(combined)  # used for scaling reference spectrum
        combined /= combined[self.spectrum_max_idx]

        # scale components for plot
        masks = [w == 0 for w in weights]

        for c, m in zip(components, masks):
            c[m] = 0

        max_val = max([np.amax(c) for c in components])
        components = [c / max_val for c in components]

        return components, combined

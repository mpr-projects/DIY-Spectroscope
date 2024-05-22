import tkinter as tk
import os
import numpy as np

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_wavelength_mapping import ViewWavelengthMapping
import helper_interpolation



class ControllerWavelengthMapping(ControllerHelper):

    def __init__(self, controller, frame, restrict=False):
        super().__init__(controller, frame)
        self.restrict = restrict

        self._set_up_variables()
        self._view = ViewWavelengthMapping(self, frame)
        self._set_up_bindings()
        self._reload_variables()

    def _set_up_variables(self):
        self._vars_nm = list()
        self._vars_px = list()

    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)
        EventHandler.bind(
            '<<ControllerLoadedState>>', self._reload_variables)

    def _reload_variables(self, *args):
        m = self.calibration.wavelength_pixel_mapping
        n_required, n_available = len(m), len(self._vars_nm)

        for _ in range(n_available, n_required):
            self.add_calibration_point(add_to_calibration=False)

        for idx in range(n_required):
            nm, px = m[idx]
            self._vars_nm[idx].set(f'{nm}nm')
            self._vars_px[idx].set(px)

        for idx in range(n_required, n_available):
            self._vars_nm[idx].set('nm')
            self._vars_px[idx].set(0)

    def handle_image_change(self):
        self._view.reset_limits()
        self._view.update_plot()

    def _clicked_tab(self, params):
        name = 'Deviation' if self.restrict else 'Wavelength-Pixel Mapping'

        if params.clicked_name != name:
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._reload_variables()
        self._view.update_plot()

    def add_calibration_point(self, add_to_calibration=True):
        v_nm = tk.StringVar(value='nm')
        self._vars_nm.append(v_nm)

        v_px = tk.StringVar()
        self._vars_px.append(v_px)

        self._view.add_calibration_point(v_nm, v_px)

        if add_to_calibration is True:
            self.calibration.wavelength_pixel_mapping.append([0, 0])

    def get_fn_calibration_point_changed(self, idx):

        def fn(*args):
            nm = self._vars_nm[idx].get()
            nm = 0 if nm == 'nm' else float(nm[:-2])

            px = self._vars_px[idx].get()
            px = 0 if px == '' else float(px)

            self.calibration.wavelength_pixel_mapping[idx] = [nm, px]
            self._view.update_plot()

        return fn

    def update_response_functions(self):
        self.calibration.update_response_functions_px_nm_mapping()

    def export_signal(self, fname=None):
        vals = self.get_average_measured_values()

        if vals is None:
            return

        if fname is None:
            ftypes = [('Binary', '*.npy'), ('Text File', '*.txt')]
            fname = tk.filedialog.asksaveasfilename(filetypes=ftypes)

        if len(fname) == 0:
            return

        save_fn = np.savetxt if fname[-3:] == 'txt' else np.save
        save_fn(fname, vals)

    def export_signal_folder(self):
        sdir = tk.filedialog.askdirectory(title='Select a Source Directory')

        if len(sdir) == 0:
            return

        odir = tk.filedialog.askdirectory(title='Select an Output Directory')

        if len(odir) == 0:
            return

        files = os.listdir(sdir)
        files_src = [os.path.join(sdir, f) for f in files]

        files_out = [
            os.path.join(odir, '.'.join(f.split('.')[:-1])+'.npy')
            for f in files
        ]

        tk.messagebox.showinfo(
            title='Starting Conversion',
            message='Conversion starts now. A message will appear when it\'s done.')

        for fs, fo in zip(files_src, files_out):
            self._c.load_picture(fpath=fs)
            self.export_signal(fname=fo)

        tk.messagebox.showinfo(
            title='Conversion Finished', message='All files have been converted.')

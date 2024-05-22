import tkinter as tk
from tkinter import filedialog

import tempfile
import json
import copy
import os
import numpy as np

from event_handler import EventHandler
from helper_camera import GetAvailableCameras, KeepCameraAwake, GPhotoImageHelper
from model_image import Image
from model_calibration import Calibration
import helper_interpolation


class Controller:
    """
    This class manages the GUI and the model. It reacts to GUI callbacks and
    affects the required changes in the model (and vice versa).
    """

    def __init__(self, view, track_changes=True, title=None):
        self.view = view
        self.calibration = Calibration()
        self._set_up_variables()
        
        self.track_changes = track_changes
        self.title = title

        self._is_taking_picture = False
        self._keep_alive = KeepCameraAwake(self)
        self._keep_alive_id = None
        self.image_loaded_callback_list = list()
        self.tempdir = tempfile.TemporaryDirectory(
            prefix='DIY_Spectroscope_Hardware_Calibration_')

    def _set_up_variables(self):
        self._var_img_info_cam = tk.StringVar()
        self._var_img_info_lens = tk.StringVar()
        self._var_img_info_fl = tk.StringVar()
        self._var_img_info_ap = tk.StringVar()
        self._var_img_info_ss = tk.StringVar()

        self._var_camera = tk.StringVar()
        self._var_auto_ss = tk.BooleanVar(value=True)
        self._var_quantile = tk.StringVar(value='99.99')

    def initialize(self, params):
        """
        First the controller is initialized with the view in function __init__.
        This creates the object and it's member functions so they are available
        as callbacks in the gui. After the gui has been built this function is
        called to finish the initialization of the program.
        """
        self.get_available_cameras()

        if params.settings_file is not None:
            self.load(fpath=params.settings_file)
        else:  # use default values of model
            EventHandler.generate('<<ControllerLoadedState>>')

        self.view.bind('<Control-s>', self.save)
        self.view.bind('<Control-o>', self.load)
        self.view.bind('<Control-n>', self.save_as)
        self.view.bind('<Control-l>', self.load_picture)
        self.view.bind('<Control-q>', self.quit)
        self.view.bind('<Control-t>', self.take_picture)

        for i in range(1, 8):
            self.view.bind(f'<Alt-KeyPress-{i}>', self.view.change_tab)

        if params.picture is not None:
            self.load_picture(fpath=params.picture)

        self.has_changed = False
        self.update_title()

    @property
    def img(self):
        return getattr(self, '_img', None)

    @property
    def has_changed(self):
        return getattr(self, '_has_changed', False)

    @has_changed.setter
    def has_changed(self, val):
        if self.track_changes is False:
            return

        self._has_changed = val
        self.update_title()

    @property
    def auto_shutterspeed(self):
        return self._var_auto_ss.get()

    @property
    def auto_shutterspeed_quantile(self):
        return self._var_quantile.get()

    @property
    def camera(self):
        cam = self._var_camera.get()
        return None if cam == '' else cam

    def load(self, *args, fpath=None):
        if fpath is None:
            fpath = filedialog.askopenfilename(
                filetypes=[('Spectrosope Settings', '*.ss')])

        if len(fpath) == 0:
            return

        with open(fpath, 'r') as f:
            data = json.load(f)

        self.calibration.load_from_dict(data)
        self.settings_file = fpath
        self.has_changed = False
        self.update_title()

        EventHandler.generate('<<ControllerLoadedState>>')

    def reload(self, *args):
        if getattr(self, 'settings_file', None) is None:
            return
        self.load(fpath=self.settings_file)

    def save(self, *args):
        file = getattr(self, 'settings_file', None)
        self.save_as(file=file)

    def save_as(self, *args, file=None):
        if file is None:
            file = filedialog.asksaveasfilename(
                filetypes=[('Spectrosope Settings', '*.ss')])

        if len(file) == 0:
            return

        self.settings_file = file

        # only save valid calibration vals
        data = self.calibration.save_to_dict()

        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

        print('Saved')
        self.has_changed = False
        self.update_title()

    def load_picture(self, *args, fpath=None):
        if fpath is None:
            idir = None if self.img is None else os.path.dirname(self.img.fpath)
            fpath = tk.filedialog.askopenfilename(initialdir=idir)

            if len(fpath) == 0:
                return

        if getattr(self, '_img', None) is not None:
            self._img.close()

        self._img = Image(self, fpath)
        self._update_image_information()
        self.notify_about_image_change()
        self.update_title()

    def _update_image_information(self):
        self._var_img_info_cam.set(f'{self.img.camera_make} - {self.img.camera_model}')
        self._var_img_info_lens.set(f'{self.img.lens_make} - {self.img.lens_model}')
        self._var_img_info_fl.set(f'{self.img.focal_length}mm')
        self._var_img_info_ap.set(f'f/{self.img.aperture}')
        self._var_img_info_ss.set(f'{self.img.shutter_speed}s')

    def notify_about_image_change(self):
        for obj in self.image_loaded_callback_list:
            obj.handle_image_change()

    def update_title(self):  # Todo: should go into the controller
        title = ''

        if self.has_changed is True:
            title += '*'

        if self.title is not None:
            title += self.title

        if getattr(self, 'settings_file', None) is not None:
            sf = os.path.basename(self.settings_file)
            title += f' - {sf}'

        if self.img is not None:
            path, name = os.path.split(self.img.fpath)
            _, path = os.path.split(path)
            fpath = os.path.join('...', path, name)
            # fpath = os.path.basename(self.img.fpath)
            title += f' - {fpath}'

        self.view.title(title)

    def monitor(self, thread, func):
        """
        Repeatedly checks if a thread is still running. Run 'func' when
        finished.
        """
        if thread.is_alive():
            self.view.after(100, lambda: self.monitor(thread, func))

        else:
            func(thread)

    def get_available_cameras(self):
        self._tmp_cameras = list()
        thread = GetAvailableCameras(self, self._tmp_cameras)
        thread.start()
        self.monitor(thread, self._get_available_cameras_finished)

    def _get_available_cameras_finished(self, thread):
        if self._is_taking_picture is True:
            self.view.after(2000, self.get_available_cameras)
            return

        self.view._cb_cameras['values'] = self._tmp_cameras

        if len(self._tmp_cameras) == 0:
            self._var_camera.set('')
            self.view.disable_camera_buttons()

        else:
            cam = self._var_camera.get()

            if cam == '' or cam not in self._tmp_cameras:
                self._var_camera.set(self._tmp_cameras[0])

            if getattr(self, 'settings_file', None) is not None:
                self.view.enable_camera_buttons()

        self.view.after(2000, self.get_available_cameras)

    def take_picture(self, *args):
        self._is_taking_picture = True
        self.view.disable_all()
        img = GPhotoImageHelper(self)
        img.start()
        self.monitor(img, self._take_picture_finished)

    def _take_picture_finished(self, img):
        if getattr(self, '_img', None) is not None:
            del self._img

        self._img = Image(self, img.fpath)
        self.notify_about_image_change()

        self.view.enable_all()
        self._is_taking_picture = False

    def keep_alive(self):
        state = self._var_keep_alive.get()

        if state == 0:  # stop keeping camera wake
            assert self._keep_alive_id is not None
            self.view.after_cancel(self._keep_alive_id)
            self._keep_alive_id = None
            return

        self._keep_alive.start()  # 100 seconds
        self._keep_alive_id = self.view.after(100000, self.keep_alive)

    def quit(self, *args):
        if getattr(self, 'has_changed', False) is True:
            q = tk.messagebox.askyesnocancel(
                title='Unsaved Changes',
                message='Do you want to save your unsaved changes?')

            if q is None:  # cancelled
                return

            if q is True:
                self.save()

        self.view.destroy()

    # functions used by multiple views
    # -------------------------------------------------------------------------
    def get_wavelengths_(self, xvals=None, return_all=False):
        m = self.calibration.wavelength_pixel_mapping
        m = [v for v in m if v[0] != 0 and v[1] != 0]

        if len(m) < 2:
            return None

        if xvals is None:
            _, _, l, r = self.calibration.rectangle_coords
            xvals = np.arange(l, r)

        m = sorted(m)
        nm = [v[0] for v in m]
        px = [v[1] for v in m]

        method = self.calibration.interpolation_method
        fn = helper_interpolation.mapping[method]
        nms = fn(px, nm, xvals)

        if return_all is True:
            return nms, xvals, [nm, px]

        return nms

    def get_wavelengths(self, xvals=None, return_all=False):
        if xvals is None:
            _, _, l, r = self.calibration.rectangle_coords
            xvals = np.arange(l, r)

        nms = self.calibration.map_px_to_nm(xvals)

        if nms is None:
            return None

        if return_all is True:
            nm, px = self.calibration.get_valid_nm_px_mappings(split=True)
            return nms, xvals, [nm, px]

        return nms

    def get_color_statistics(self):
        if self.img is None:
            return None

        t, b, l , r = self.calibration.rectangle_coords
        tl, br = [l, t], [r, b]
        stats = self._img.get_statistics(tl=tl, br=br)
        xvals = np.arange(l, r)
        return stats, xvals

    def get_valid_mappings(self):
        m = self.calibration.wavelength_pixel_mapping
        v = list()

        for nm, px in m:
            if nm != 0 and px != 0:
                v.append([nm, px])

        return v

    def get_average_measured_values(self):
        # need at least two lines to map pixels to wavelengths
        if self.img is None:
            return None

        m = self.get_valid_mappings()

        if len(m) < 2:
            return None

        stats, xvals = self.get_color_statistics()
        nms = self.get_wavelengths(xvals)

        bl_r, bl_g, bl_b = self.calibration.black_levels
        wl = self._img.white_level

        r = (stats['mean_red']-bl_r) / (wl-bl_r)
        g = (stats['mean_green']-bl_g) / (wl-bl_g)
        b = (stats['mean_blue']-bl_b) / (wl-bl_b)

        r = np.maximum(r, 0)
        g = np.maximum(g, 0)
        b = np.maximum(b, 0)

        return nms, r, g, b

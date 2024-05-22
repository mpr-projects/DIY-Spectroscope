# inspired by
#  - https://www.pythontutorial.net/tkinter/tkinter-thread/
import os
import re
import argparse
import numpy as np

import tkinter as tk
from tkinter import ttk

from event_handler import EventHandler
from controller import Controller
from controller_picture import ControllerPicture
from controller_single_wavelength import ControllerSingleWavelength
from controller_wavelength_mapping import ControllerWavelengthMapping
from controller_wavelength_interpolation import ControllerWavelengthInterpolation
from controller_spectral_response import ControllerSpectralResponse
from controller_response_mixing import ControllerResponseMixing
from controller_spectrum import ControllerSpectrum
import helper_interpolation
from helper_misc import is_float


# np.seterr(all='raise')


class Spectroscope_HC(tk.Tk):

    def __init__(self, params=None):
        super().__init__()
        EventHandler.root = self

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # widgets draw in here
        self._r = ttk.Frame(self)
        self._r.grid(row=0, column=0, sticky='nesw', padx=10, pady=10)

        self._r.rowconfigure(1, weight=1)
        self._r.columnconfigure(0, weight=1)

        # the controller manages view (this class) and model; it needs to be
        # instantiated before the gui is built so the various callback
        # functions of the gui are available
        self.c = Controller(
            self, title='DIY Spectroscope - Hardware Calibration')

        # build interface
        self._build_top()
        self._build_bottom()

        # finish initializing the controller
        self.c.initialize(params)

    def _build_top(self):
        c = self.c

        f = self._f = ttk.Frame(self._r)
        f.grid(row=0, column=0, sticky='nesw')

        f.columnconfigure(6, weight=1)

        # settings file
        ttk.Label(f, text='Settings File').grid(row=0, column=0, sticky='w')

        f_sf = ttk.Frame(f)
        f_sf.grid(row=0, column=1, sticky='w')

        ttk.Button(f_sf, text='Open', underline=0, command=c.load).grid(
            row=0, column=0, sticky='w', padx=(0, 4))

        ttk.Button(
            f_sf, text='Save', command=c.save, underline=0
        ).grid(row=0, column=1, sticky='w', padx=4)

        ttk.Button(
            f_sf, text='Save As', underline=0, command=c.save_as
        ).grid(row=0, column=2, sticky='w', padx=4)

        def main_program():
            c.save()
            sf = getattr(self.c, 'settings_file', None)
            cmd = 'python view.py'

            if sf is not None:
                cmd += f' --settings_file {sf}'

            if self.c.img is not None:
                cmd += f' --picture {self.c.img.fpath}'

            self.destroy()
            # cmd = 'python_environment/bin/' + cmd
            # print('cmd:', cmd)
            os.system(cmd)

        self._btn_main = ttk.Button(
            f, text='Main Program', command=main_program)
        # self._btn_main.grid(row=0, column=5, sticky='e')

        ttk.Label(f, text='Load Image').grid(row=1, column=0, sticky='w')
        ttk.Label(f, text='From File').grid(
            row=2, column=0, sticky='w', padx=(20, 5))

        self._btn_load_image = b = ttk.Button(
            f, text='Load', command=c.load_picture, underline=0)
        b.grid(row=2, column=1, sticky='w')

        # camera details
        ttk.Label(f, text='From Camera').grid(
            row=3, column=0, sticky='w', padx=(20, 5))

        f_cd = ttk.Frame(f)
        f_cd.grid(row=3, column=1, columnspan=3, sticky='w')

        self._cb_cameras = cb = ttk.Combobox(
            f_cd, textvariable=c._var_camera)
        cb.grid(row=0, column=0, sticky='nsw')
        cb.state(['readonly', 'disabled'])

        self._cb_ss = cb = ttk.Checkbutton(
            f_cd, text='Auto-Shutterspeed', variable=c._var_auto_ss)
        cb.grid(row=0, column=1, sticky='nsw', padx=(10, 2))
        cb.state(['disabled'])

        self._lbl_quantile = l = ttk.Label(f_cd, text='Quantile:')
        l.grid(row=0, column=2, sticky='nsw', padx=2)
        l.state(['disabled'])

        validate = (f.register(is_float), '%P')

        self._e_quantile = e = ttk.Entry(
            f_cd, textvariable=c._var_quantile, width=5,
            validate='key', validatecommand=validate
        )
        e.grid(row=0, column=3, sticky='nsw', padx=2)
        e.state(['disabled'])

        self._btn_take = b = ttk.Button(
            f_cd, text='Take', command=c.take_picture, underline=0)
        b.grid(row=0, column=4, sticky='nsw', padx=(10, 0))
        b.state(['disabled'])

        # image information
        self._f_image_information = f_ii = ttk.Frame(f)
        self.c.image_loaded_callback_list.append(self)

        ttk.Label(f_ii, text='Image Information').grid(
            row=0, column=0, columnspan=2, pady=(0, 4), sticky='w')

        ttk.Label(f_ii, text='Camera').grid(
            row=1, column=0, padx=(10, 0), pady=2, sticky='w')

        ttk.Label(f_ii, textvariable=c._var_img_info_cam).grid(
            row=1, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(f_ii, text='Lens').grid(
            row=2, column=0, padx=(10, 0), pady=2, sticky='w')

        ttk.Label(f_ii, textvariable=c._var_img_info_lens).grid(
            row=2, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(f_ii, text='Focal Length').grid(
            row=3, column=0, padx=(10, 0), pady=2, sticky='w')

        ttk.Label(f_ii, textvariable=c._var_img_info_fl).grid(
            row=3, column=1, padx=5, pady=2, sticky='w')
            
        ttk.Label(f_ii, text='Aperture').grid(
            row=4, column=0, pady=2, padx=(10, 0), sticky='w')

        ttk.Label(f_ii, textvariable=c._var_img_info_ap).grid(
            row=4, column=1, pady=2, padx=5, sticky='w')
            
        ttk.Label(f_ii, text='Shutter Speed').grid(
            row=5, column=0, padx=(10, 0), pady=2, sticky='w')

        ttk.Label(f_ii, textvariable=c._var_img_info_ss).grid(
            row=5, column=1, padx=5, pady=2, sticky='w')
            
        # separator
        ttk.Separator(f, orient='vertical').grid(
            row=0, column=5, rowspan=4, sticky='nesw', pady=2, padx=(15, 5))

        ttk.Separator(f, orient='horizontal').grid(
            row=4, column=0, columnspan=9, sticky='nesw', pady=(10, 5))

        # padding
        for row in range(3):
            f.rowconfigure(row, pad=10)

        f.rowconfigure(4, pad=5)

        for col in range(9):
            f.columnconfigure(col, pad=5)

    def _build_bottom(self):
        l = self.c.image_loaded_callback_list

        self._nb = ttk.Notebook(self._r)
        self._nb.grid(row=1, column=0, sticky='nesw', pady=0)

        # I'm using this instead of <<NotebookTabChanged>> because I need
        # to know both the old and the new tab
        self._nb.bind('<Button-1>', self._nb_tab_click_notify)

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Picture')
        l.append(ControllerPicture(self.c, nb))

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Single Wavelength')
        l.append(ControllerSingleWavelength(self.c, nb))

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Wavelength-Pixel Mapping')
        l.append(ControllerWavelengthMapping(self.c, nb))

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Wavelength Interpolation')
        ControllerWavelengthInterpolation(self.c, nb)

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Spectral Response')
        l.append(ControllerSpectralResponse(self.c, nb))

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Response Mixing')
        ControllerResponseMixing(self.c, nb)

        nb = ttk.Frame(self._nb)
        self._nb.add(nb, text='Spectrum')
        l.append(ControllerSpectrum(self.c, nb))

    def handle_image_change(self):
        ii = self._f_image_information

        if not ii.winfo_ismapped():
            ii.grid(row=0, column=6, rowspan=4, sticky='w')

    def change_tab(self, event):
        tab_idx = int(event.char) - 1

        if tab_idx > 7:
            return

        tab_name = self._nb.tab(tab_idx, option='text')
        active_tab_idx = self._nb.index(self._nb.select())
        self._nb.select(tab_idx)

        EventHandler.generate(
            f'<<Clicked_Tab>>', clicked_idx=tab_idx,
            clicked_name=tab_name, active_idx=active_tab_idx)

    def _nb_tab_click_notify(self, event):
        tab_idx = self._nb.tk.call(
            self._nb._w, 'identify', 'tab', event.x, event.y)

        # clicking just below the name of the tab causes this function to
        # be called but not tab idx is given, in that case we just return
        if tab_idx == '':
            return

        tab_name = self._nb.tab(tab_idx, option='text')
        active_tab_idx = self._nb.index(self._nb.select())

        if self._nb.tab(tab_idx, option='state') == 'disabled':
            return

        EventHandler.generate(
            f'<<Clicked_Tab>>', clicked_idx=tab_idx,
            clicked_name=tab_name, active_idx=active_tab_idx)

    def enable_camera_buttons(self):  # enabled when camera is connected
        self._btn_take.state(['!disabled'])
        self._cb_ss.state(['!disabled'])
        self._e_quantile.state(['!disabled'])
        self._lbl_quantile.state(['!disabled'])
        self._cb_cameras.state(['readonly', '!disabled'])

    def disable_camera_buttons(self):  # disabled when camera is disconnected
        self._btn_take.state(['disabled'])
        self._cb_ss.state(['disabled'])
        self._e_quantile.state(['disabled'])
        self._lbl_quantile.state(['disabled'])
        self._cb_cameras.state(['readonly', 'disabled'])

    def enable_all(self):
        self._btn_calibrate.state(['!disabled'])
        self._btn_load_image.state(['!disabled'])
        self.enable_camera_buttons()

    def disable_all(self):
        self._btn_calibrate.state(['disabled'])
        self._btn_load_image.state(['disabled'])
        self.disable_camera_buttons()


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Spectroscope Hardware Calibration',
        description=(
            'Calibrate the hardware side of your camera for the'
            ' DIY Spectroscope.'))

    parser.add_argument('--settings_file',
                        help='Path to newly created settings file.')

    parser.add_argument('--picture',
                        help='Pictures on disk to load.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    app = Spectroscope_HC(params=args)
    app.mainloop()

from threading import Thread
import numpy as np
import os
import copy
import shutil
import datetime
import subprocess
import signal
import tempfile
import rawpy
import lensfunpy
import pyexiv2
import cv2

import tkinter as tk
from tkinter import messagebox

from event_handler import EventHandler


class Image:

    def __init__(self, controller, fpath):
        self._c = controller
        self._fpath = fpath
        self._raw = raw = rawpy.imread(fpath)
        self._check_for_xtrans()
        self.load_metadata()
        self.apply_lens_corrections()

    @property
    def fpath(self):
        return self._fpath

    @property
    def raw_image(self):
        ra = getattr(self, '_raw_image_adj', None)

        if ra is None:
            return self._raw.raw_image_visible

        return ra

    @property
    def raw_colors(self):
        rc = getattr(self, '_raw_colors_adj', None)

        if rc is None:
            return self._raw_colors_visible

        return rc

    @property
    def white_level(self):
        return self._raw.white_level

    def load_metadata(self):
        md = pyexiv2.ImageMetadata(self._fpath)
        md.read()

        cmake = md.get('Exif.Image.Make')
        cmodel = md.get('Exif.Image.Model')

        cmake = None if cmake is None else cmake.value
        cmodel = None if cmodel is None else cmodel.value

        lmake = md.get('Exif.Photo.LensMake')
        lmodel = md.get('Exif.Photo.LensModel')

        lmake = None if lmake is None else lmake.value
        lmodel = None if lmodel is None else lmodel.value

        focal_length = md.get_focal_length()
        aperture = md.get_aperture()
        ss = md.get_shutter_speed(float_=True)

        self.camera_make = cmake
        self.camera_model = cmodel
        self.lens_make = lmake
        self.lens_model = lmodel
        self.focal_length = focal_length
        self.aperture = aperture
        self.shutter_speed = ss

    def get_exiv_metadata(self):
        return (self.camera_make, self.camera_model,
                self.lens_make, self.lens_model,
                self.focal_length, self.aperture)

    @property
    def filetype(self):
        return self._fpath.split('.')[-1]

    def _check_for_xtrans(self):
        xtrans =  np.array([
            [1, 1, 0, 1, 1, 2],
            [1, 1, 2, 1, 1, 0],
            [2, 0, 1, 0, 2, 1],
            [1, 1, 2, 1, 1, 0],
            [1, 1, 0, 1, 1, 2],
            [0, 2, 1, 2, 0, 1]])

        if not (self._raw.raw_pattern == xtrans).all():
            raise NotImplementedError(
                'Currently only X-Trans (Fuji) sensors are supported')

    def save_as(self, fpath):
        shutil.copy2(self._fpath, fpath)

    def apply_lens_corrections(self):
        self._raw_image_adj = self._raw.raw_image_visible
        self._raw_colors_adj = self._raw.raw_colors_visible

        # I'm not keeping track of the previous value of vignetting, tca and
        # distortion; we could potentially save some computation by doing so
        # but we have to take the (fixed) order of the adjustments into account,
        # I don't think it's worth it ...

        # reset statistics so they are recomputed when required
        self._reset_statistics()

        vignetting, tca, distortion = self._c.calibration.lens_corrections

        if not any([vignetting, tca, distortion]):
            return

        exif = self.get_exiv_metadata()

        if any([e is None for e in exif]):
            # Todo: send an error message
            return  # not all required data could be read

        (cam_make, cam_model, lens_make, lens_model,
         focal_length, aperture) = exif

        db = lensfunpy.Database()
        cam = db.find_cameras(cam_make, cam_model)

        if len(cam) == 0:
            msg = 'Camera not found in lensfun database. No corrections applied.'
            tk.messagebox.showerror(title='Error', message=msg)
            EventHandler.generate('<<ResetLensCorrections>>')
            return

        cam = cam[0]
        lens = db.find_lenses(cam, lens_make, lens_model)

        if len(lens) == 0:
            msg = 'Lens not found in lensfun database. No corrections applied.'
            tk.messagebox.showerror(title='Error', message=msg)
            EventHandler.generate('<<ResetLensCorrections>>')
            return

        lens = lens[0]

        focus_distance = 0.5  # should be similar for different spectroscopes

        height, width = self._raw.raw_image_visible.shape
        mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
        mod.initialize(focal_length, aperture, 0.5, pixel_format=np.float64) 

        # help(mod.initialize)
        # help(lensfunpy.ModifyFlags)

        if vignetting is True:
            self._apply_devignetting(mod)

        if tca is True:
            self._apply_tca(mod)

        if distortion is True:
            self._apply_distortion(mod)

        self._raw_image_adj = np.maximum(self._raw_image_adj, 0)
        self._raw_image_adj = np.minimum(self._raw_image_adj, self.white_level)

    def _apply_devignetting(self, mod):
        height, width = self._raw.raw_image_visible.shape
        img = np.ones((height, width, 3), dtype=np.float64)
        did_apply = mod.apply_color_modification(img)

        if did_apply is False:
            tk.messagebox.showerror(
                title='Error', message='Couldn\'t apply devignetting.')
            return

        # need to account for black level when correcting vignetting
        bls = self._c.calibration.black_levels
        colors = self._raw_colors_adj

        raw_adj = copy.deepcopy(self._raw_image_adj).astype(np.float32)

        for i in range(3):
            raw_adj[colors == i] -= bls[i]

        # vignetting should be the same for all colors, doesn't matter if we
        # pick 0, 1 or 2 in the last dimension
        raw_adj *= img[..., 0]

        for i in range(3):
            raw_adj[colors == i] += bls[i]

        raw_adj = np.maximum(raw_adj, 0)
        raw_adj = np.round(raw_adj)
        self._raw_image_adj = raw_adj.astype(self._raw.raw_image_visible.dtype)

    def _apply_tca(self, mod):
        undist_coords = mod.apply_subpixel_distortion()

        colors, values = list(), list()

        # I'm using nearest neighbour interpolation because for the other
        # types of interpolation we'd need a full grid for each color channel;
        # we don't have information about all colors at all pixels because that
        # would require demosaicing; however, we don't demosaic the image
        # because that would mix the data from different wavelengths

        # as a result both tca and distortion adjustments may not be perfectly
        # accurate

        for i in range(3):
            c = cv2.remap(
                    self._raw_colors_adj,
                    undist_coords[:, :, i, :],
                    None, cv2.INTER_NEAREST)

            v = cv2.remap(
                    self._raw_image_adj,
                    undist_coords[:, :, i, :],
                    None, cv2.INTER_NEAREST)

            colors.append(c)
            values.append(v)

        
        # multiple colors could map onto the same pixel; if we were saving the
        # data in three separate channels (as would be done after demosaicing)
        # then we wouldn't lose any information; however, we save all color
        # data in one channel so if two colors are mapped to the same pixel we
        # get a clash; I resolve that clash by only keeping one of the colors
        # (red > green > blue);

        # I did a test to see if it's worth saving the three color channels
        # separately; I don't think it is because almost no pixels had two
        # colors mapped to them (so we don't lose much information) but the
        # program became even slower

        colors = np.stack(colors, axis=-1)
        ol = np.logical_or(colors[...,0] != colors[..., 1], colors[..., 1] != colors[..., 2])

        colors = np.min(colors, axis=-1, keepdims=True)

        values = np.stack(values, axis=-1)
        values = np.take_along_axis(values, colors, axis=-1)

        self._raw_colors_adj = colors[..., 0]
        self._raw_image_adj = values[..., 0]

    def _apply_distortion(self, mod):
        undist_coords = mod.apply_geometry_distortion()

        self._raw_image_adj = cv2.remap(
            self._raw_image_adj, undist_coords, None, cv2.INTER_NEAREST)

        self._raw_colors_adj = cv2.remap(
            self._raw_colors_adj, undist_coords, None, cv2.INTER_NEAREST)

    def get_image(self):
        """ Returns an image that can be plotted. """
        raw = self.raw_image

        # assuming that we're well exposed (there are bright areas in the image)
        # max_val = 2**np.ceil(np.log2(np.amax(raw))) - 1
        max_val = self.white_level
        raw_vals = raw / max_val

        img_shape = raw_vals.shape + (3,)
        img = np.zeros(img_shape)

        for i in range(3):
            mask_1 = (self.raw_colors == i)
            mask_2 = np.zeros(3, dtype=bool)
            mask_2[i] = True
            mask = np.logical_and(mask_1[:, :, None], mask_2[None, None, :])

            img[mask] = raw_vals[mask_1]

        return img

    def _reset_statistics(self):
        self._statistics = None
        self._statistics_tl = None
        self._statistics_br = None

    def get_statistics(self, tl=None, br=None):
        """
        Returns mean and std of each color in each column in the rectangle
        defined by the top left (tl) and bottom right (br) coordinates. If not
        given then the entire image will be used.
        """

        # check if we've already computed the statistics
        s1 = getattr(self, '_statistics', None) is not None
        s2 = getattr(self, '_statistics_tl', None) == tl
        s3 = getattr(self, '_statistics_br', None) == br

        if s1 and s2 and s3:
            return self._statistics

        colors = ['red', 'green', 'blue']  # Todo: get from rawpy
        res = dict()

        vals = self.raw_image
        cols = self.raw_colors

        tl = [0, 0] if tl is None else tl
        br = vals.shape if br is None else br

        vals = vals[tl[1]:br[1], tl[0]:br[0]]
        cols = cols[tl[1]:br[1], tl[0]:br[0]]

        for i in range(3):
            res[f'mean_{colors[i]}'] = np.mean(vals, axis=0, where=(cols==i))
            res[f'std_{colors[i]}'] = np.std(vals, axis=0, where=(cols==i))

        self._statistics = res
        self._statistics_tl = tl
        self._statistics_br = br

        return res

    def close(self):
        self._raw.close()

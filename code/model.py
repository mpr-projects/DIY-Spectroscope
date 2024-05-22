from threading import Thread
import numpy as np
import os
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
        self._convert_to_RGB()
        self.apply_lens_corrections(send_event=False)
        self._update_state()

    @property
    def fpath(self):
        return self._fpath

    @property
    def raw_image(self):
        ra = getattr(self, '_raw_image_adj', None)

        if ra is None:
            return self._raw.raw_image

        return ra

    @property
    def raw_colors(self):
        rc = getattr(self, '_raw_colors_adj', None)

        if rc is None:
            return self._raw_colors

        return rc

    """
    @property
    def color_desc(self):
        return self._raw.color_desc.decode('ascii')
    """

    @property
    def white_level(self):
        return self._raw.white_level

    def get_exiv_metadata(self):
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

        return cmake, cmodel, lmake, lmodel, focal_length, aperture

    def _update_state(self):
        (cam_make, cam_model, lens_make, lens_model,
         focal_length, aperture) = self.get_exiv_metadata()

        keys = ['cam_make', 'cam_model', 'lens_make', 'lens_model',
                'focal_length', 'aperture']

        EventHandler.generate(
            '<<UpdateState>>', keys=keys, cam_make=cam_make,
            cam_model=cam_model, lens_make=lens_make, lens_model=lens_model,
            focal_length=float(focal_length), aperture=float(aperture))

    @property
    def filetype(self):
        return self._fpath.split('.')[-1]

    def _convert_to_RGB(self):
        # this code only supports RGB, if the source image is in RGBG
        # then we convert it to RGB
        colors = self._raw.color_desc.decode('ascii')

        if colors == 'RGB':
            self._raw_colors = self._raw.raw_colors
            return

        if colors != 'RGBG':
            # Todo: send a tk message
            raise RuntimeError(
                'Only RGB is supported. RGBG is converted to RGB.'
                f' {colors} is not supported.')

        self._raw_colors = np.where(
            self._raw.raw_colors == 3, 1, self._raw.raw_colors)

    def save_as(self, fpath):
        shutil.copy2(self._fpath, fpath)

    def apply_lens_corrections(self, send_event=True):
        print('APPLYING LENS CORRECTION!')
        self._raw_image_adj = self._raw.raw_image
        self._raw_colors_adj = self._raw_colors

        # I'm not keeping track of the previous value of vignetting, tca and
        # distortion; we could potentially save some computation by doing so
        # but we have to take the (fixed) order of the adjustments into account,
        # I don't think it's worth it ...

        # reset statistics so they are recomputed when required
        self._reset_statistics()

        """
        vignetting = self._c.apply_vignetting
        tca = self._c.apply_tca
        distortion = self._c.apply_distortion
        """

        print('calib:', self._c.calibration.lens_corrections)
        vignetting, tca, distortion = self._c.calibration.lens_corrections

        if not any([vignetting, tca, distortion]):
            print('no lens corrections, returning')
            if send_event is True:
                EventHandler.generate('<<RAW_Image_Updated>>')

            EventHandler.generate('<<Applied_Lens_Corrections>>')
            return

        exif = self.get_exiv_metadata()

        if any([e is None for e in exif]):
            # Todo: send an error message
            return  # not all required data could be read

        (cam_make, cam_model, lens_make, lens_model,
         focal_length, aperture) = exif

        db = lensfunpy.Database()

        cam = db.find_cameras(cam_make, cam_model)[0]
        lens = db.find_lenses(cam, lens_make, lens_model)[0]

        focus_distance = 0.5  # should be similar for different spectroscopes

        height, width = self._raw.raw_image.shape
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

        if send_event is True:
            EventHandler.generate('<<RAW_Image_Updated>>')

        EventHandler.generate('<<Applied_Lens_Corrections>>')

    def _apply_devignetting(self, mod):
        height, width = self._raw.raw_image.shape
        img = np.ones((height, width, 3), dtype=np.float64)
        did_apply = mod.apply_color_modification(img)

        if did_apply is False:
            tk.messagebox.showerror(
                title='Error', message='Couldn\'t apply devignetting.')
            return

        """  some tests, can't reconcile the values I get here with the coefficients
             in the lensfun database, there is some hugin-related rescaling done in
             lensfun's code, don't know what this is doing ...
        xdist = 1000
        print('vig img:', img[height//2, width//2, 0], img[height//2, width//2-xdist, 0])
        print('  dist:', width//2, width//2-xdist)
        diag = (width**2 + height**2)**0.5
        print('  diag:', diag, diag/2)

        vs = img[height//2, width//2:, 0]
        import matplotlib.pyplot as plt
        plt.plot(vs)
        plt.show()
        """

        raw_adj = np.round(self._raw_image_adj * img[..., 0])
        self._raw_image_adj = raw_adj.astype(self._raw.raw_image.dtype)

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
        # print('# overlapping pixels =', ol.sum())

        colors = np.min(colors, axis=-1, keepdims=True)

        values = np.stack(values, axis=-1)
        values = np.take_along_axis(values, colors, axis=-1)

        self._raw_colors_adj = colors[..., 0]
        self._raw_image_adj = values[..., 0]

    def _apply_distortion(self, mod):
        undist_coords = mod.apply_geometry_distortion()

        """ distortion seems to be pretty much linear with distance from center
        print('shape undist_coords:', undist_coords.shape)
        vs = undist_coords[4038//2, 6048//2:]
        import matplotlib.pyplot as plt
        plt.plot(vs[:, 0], label='0')
        plt.plot(vs[:, 1], label='1')
        plt.legend()
        plt.show()
        raise RuntimeError()
        """

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
        print('white level:', self.white_level, np.amax(raw))
        raw_vals = raw / max_val

        img_shape = raw_vals.shape + (3,)
        img = np.zeros(img_shape)

        for i in range(3):
            mask_1 = (self.raw_colors == i)
            mask_2 = np.zeros(3, dtype=bool)
            mask_2[i] = True
            mask = np.logical_and(mask_1[:, :, None], mask_2[None, None, :])

            img[mask] = raw_vals[mask_1]

        """
        import matplotlib.pyplot as plt
        plt.imshow(self.raw_image)
        plt.show()

        plt.imshow(self.raw_colors)
        plt.show()
        """
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

        """
        import matplotlib.pyplot as plt
        plt.imshow(vals)
        plt.show()

        plt.imshow(cols==0)
        plt.show()
        """

        for i in range(3):
            print('i', i)
            res[f'mean_{colors[i]}'] = np.mean(vals, axis=0, where=(cols==i))
            res[f'std_{colors[i]}'] = np.std(vals, axis=0, where=(cols==i))

        """
        plt.plot(res[f'mean_{colors[0]}'])
        plt.show()
        """
        # raise RuntimeError()

        self._statistics = res
        self._statistics_tl = tl
        self._statistics_br = br

        return res

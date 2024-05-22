import tkinter as tk
from tkinter import filedialog
from PIL import Image as PIL_Image

import os
import numpy as np

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_picture import ViewPicture
from helper_misc import px


class ControllerPicture(ControllerHelper):
    """ This class is the controller for the 'Picture' view. """

    def __init__(self, controller, frame, restrict=False):
        super().__init__(controller, frame)
        self._restrict = restrict

        self.canvas_width, self.canvas_height = None, None

        self._set_up_variables()
        self._view = ViewPicture(self, frame)
        self._set_up_bindings()

        # this class has two 'images', one is the property 'img', which
        # refers to the image object held by the main controller, the
        # other is '_img' which is a PIL image shown in this tab

    def _set_up_variables(self):
        # Global
        self._var_rect_top = tk.IntVar()
        self._var_rect_bottom = tk.IntVar()
        self._var_rect_left = tk.IntVar()
        self._var_rect_right = tk.IntVar()
        self._var_dashed = tk.IntVar()

        self._var_global_bl_r = tk.IntVar()
        self._var_global_bl_g = tk.IntVar()
        self._var_global_bl_b = tk.IntVar()

        self._var_vignetting = tk.BooleanVar()
        self._var_tca = tk.BooleanVar()
        self._var_distortion = tk.BooleanVar()

        # Tab 'Picture'
        self._var_show_r = tk.BooleanVar()
        self._var_show_g = tk.BooleanVar()
        self._var_show_b = tk.BooleanVar()

        self._var_bl_r = tk.IntVar(value=1300)
        self._var_bl_g = tk.IntVar(value=1600)
        self._var_bl_b = tk.IntVar(value=1600)

        self._var_mouse_info = tk.StringVar()

    def _set_up_bindings(self):
        self._f.bind('<Configure>', self._resize)
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)

        EventHandler.bind(
            '<<ControllerLoadedState>>', self._loaded_state_callback)

        EventHandler.bind(
            '<<ResetLensCorrections>>', self.reset_lens_corrections)

    def _reload_variables(self):
        rect = self.calibration.rectangle_coords
        self._var_rect_top.set(rect[0])
        self._var_rect_bottom.set(rect[1])
        self._var_rect_left.set(rect[2])
        self._var_rect_right.set(rect[3])

        dashed = self.calibration.dashed_coord
        self._var_dashed.set(dashed)

        r, g, b = self.calibration.black_levels

        self._var_global_bl_r.set(r)
        self._var_global_bl_g.set(g)
        self._var_global_bl_b.set(b)

        v, t, d = self.calibration.lens_corrections

        self._var_vignetting.set(v)
        self._var_tca.set(t)
        self._var_distortion.set(d)

        r, g, b = self.calibration.tab_picture['show_channels']

        self._var_show_r.set(r)
        self._var_show_g.set(g)
        self._var_show_b.set(b)

        r, g, b = self.calibration.tab_picture['black_levels']

        self._var_bl_r.set(r)
        self._var_bl_g.set(g)
        self._var_bl_b.set(b)

    def _loaded_state_callback(self, *args):
        self._reload_variables()

        # lens corrections might have changed, just reload the image
        if self.img is not None:
            self.handle_image_change(reset_view=False)

    def _clicked_tab(self, params):
        if params.clicked_name != 'Picture':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._reload_variables()

        if self.img is not None:
            self._view.draw_rect()

    def _update_root_size(self):
        self._c.view.update()  # Todo: improve this
        w, h = self._view.get_canvas_size()
        self.width_root, self.height_root = w, h

    def _resize(self, *args, redraw=True):
        if self.img is None:
            return

        self._update_root_size()
        wr, hr = self.width_root, self.height_root
        w, h = self._img.size

        r = min(wr / w, hr / h)
        wc, hc = r * w, r * h  # canvas width and height

        if wc == self.canvas_width and hc == self.canvas_height:
            return

        self.canvas_width, self.canvas_height = wc, hc
        self._view.resize(wc, hc)

        if redraw is True:
            self._view.draw()

    def _reset_rect(self):
        if self.img is None:
            return

        w, h = self._img.size
        l, t = px(w / 4, h / 4)
        r, b = px(w / 4 * 3, h / 4 * 3)

        self._var_rect_top.set(t)
        self._var_rect_bottom.set(b)
        self._var_rect_left.set(l)
        self._var_rect_right.set(r)
        self._var_dashed.set( (l+r)//2 )

        self.handle_rect_coord_change(redraw=False)
        self.handle_dashed_coord_change(redraw=False)

    def handle_image_change(self, reset_view=True):
        """
        This function is called by the main controller when an image has been
        loaded. It is also called after applying lens corrections.
        """
        # PIL needs 8-bit integers
        img = self.img.get_image()
        img = (img * 255).astype(np.uint8)
        img = PIL_Image.fromarray(img)
        self._img = img.convert('RGBA')

        if reset_view is True:
            self._view_img = [0, 0, *img.size]

        rc = self.calibration.rectangle_coords
        dc = self.calibration.dashed_coord

        if rc == [0, 0, 0, 0] and dc == 0:
            self._reset_rect()

        self._resize(redraw=False)
        self._update_alpha()
        self._view.draw()

    def handle_rect_coord_change(self, *args, redraw=True):
        c = [self._var_rect_top.get(),
             self._var_rect_bottom.get(),
             self._var_rect_left.get(),
             self._var_rect_right.get()]

        if c == self.calibration.rectangle_coords:
            return

        self._c.has_changed = True
        self.calibration.rectangle_coords = c

        if redraw is True:
            self._view.draw_rect()

    def handle_dashed_coord_change(self, *args, redraw=True):
        c = self._var_dashed.get()

        if c == self.calibration.dashed_coord:
            return

        l = self._var_rect_left.get()
        r = self._var_rect_right.get()

        if not (c > l and c < r):
            c = (l + r) // 2
            self._var_dashed.set(c)

        self._c.has_changed = True
        self.calibration.dashed_coord = c

        if redraw is True:
            self._view.draw_rect()

    def handle_global_black_levels_change(self, *args):
        b = [self._var_global_bl_r.get(),
             self._var_global_bl_g.get(),
             self._var_global_bl_b.get()]

        if b == self.calibration.black_levels:
            return

        self._c.has_changed = True
        self.calibration.black_levels = b

        # if local black levels are are 0, update them too
        c = [self._var_bl_r,
             self._var_bl_g,
             self._var_bl_b]

        if all([v.get() == 0 for v in c]):
            for l, g in zip(c, b):
                l.set(g)

            self.handle_local_black_levels_change()

    def have_lens_corrections_changed(self):
        l = [self._var_vignetting.get(),
             self._var_tca.get(),
             self._var_distortion.get()]

        return l != self.calibration.lens_corrections

    def handle_lens_correction_change(self):
        l = [self._var_vignetting.get(),
             self._var_tca.get(),
             self._var_distortion.get()]

        if l == self.calibration.lens_corrections:
            return

        self._c.has_changed = True
        self.calibration.lens_corrections = l

        if self.img is not None:
            self.img.apply_lens_corrections()
            self.handle_image_change()

    def reset_lens_corrections(self, *args):
        self._var_vignetting.set(False)
        self._var_tca.set(False)
        self._var_distortion.set(False)
        self._view.lens_correction_update()

    def handle_show_channel_change(self):
        s = [self._var_show_r.get(),
             self._var_show_g.get(),
             self._var_show_b.get()]

        if s == self.calibration.tab_picture['show_channels']:

            return

        self._c.has_changed = True
        self.calibration.tab_picture['show_channels'] = s

        if self.img is not None:
            self._update_alpha()
            self._view.draw()

    def handle_local_black_levels_change(self, *args):
        b = [self._var_bl_r.get(),
             self._var_bl_g.get(),
             self._var_bl_b.get()]

        if b == self.calibration.tab_picture['black_levels']:
            return

        self._c.has_changed = True
        self.calibration.tab_picture['black_levels'] = b

        if self.img is not None:
            self._update_alpha()
            self._view.draw()

    def _update_alpha(self):
        raw = self.img.raw_image
        col = self.img.raw_colors
        alpha = np.zeros_like(raw, dtype=np.uint8)

        # hide colors that shouldn't be shown
        show = [self._var_show_r, self._var_show_g, self._var_show_b]
        bls = [self._var_bl_r, self._var_bl_g, self._var_bl_b]

        for i in range(3):
            # only show colors selected in ibar
            if show[i].get() is False:
                continue

            # only show pixels whose value is above the threshold
            alpha = np.where(
                np.logical_and(col==i, raw>=bls[i].get()), 255, alpha)

        alpha = PIL_Image.fromarray(alpha, mode='L')
        self._img.putalpha(alpha)

    def save_image_as(self):
        if self.img is None:
            return

        ftype = self.img.filetype

        file = filedialog.asksaveasfilename(
            filetypes=[('Raw Image', f'*.{ftype}')])

        if len(file) == 0:
            return

        self.img.save_as(file)

    def image_coords(self, x, y):
        assert self.img is not None
        v = self._view_img

        if self.canvas_width is None:
            self._resize(redraw=False)

        x = v[0] + x / self.canvas_width * (v[2] - v[0])
        y = v[1] + y / self.canvas_height * (v[3] - v[1])
        return x, y

    def canvas_coords(self, x=None, y=None):
        assert self.img is not None
        v = self._view_img

        if self.canvas_width is None:
            self._resize(redraw=False)

        if x is not None:
            x = (x - v[0])/(v[2] - v[0]) * self.canvas_width

        if y is not None:
            y = (y - v[1])/(v[3] - v[1]) * self.canvas_height

        return x, y

    def update_mouse_info(self, event):
        vi = getattr(self, '_view_img', None)

        if vi is None or self.img is None:
            return

        mx = self._view.canvas.canvasx(event.x)
        my = self._view.canvas.canvasy(event.y)

        ix, iy = px(*self.image_coords(mx, my))

        # moving the mouse from the outside into the
        # canvas sometimes results in larger values
        width, height = self._img.size
        ix = max(0, min(width-1, ix))
        iy = max(0, min(height-1, iy))

        val = self.img.raw_image[iy, ix]
        col = self.img.raw_colors[iy, ix]
        col = ['R', 'G', 'B'][col]

        s = f'Photosite X={ix} Y={iy}: {col} {val}'
        self._var_mouse_info.set(s)

    def get_rect_coords(self, canvas=True):
        """
        Provides the rectangle coordinates. If canvas is False then the
        coordinates are returned in image space, other wise in canvas space.
        """
        assert self.img is not None
        t, b, l, r = self.calibration.rectangle_coords

        if canvas is True:
            t = self.canvas_coords(y=t)[1]
            b = self.canvas_coords(y=b)[1]
            l = self.canvas_coords(x=l)[0]
            r = self.canvas_coords(x=r)[0]
            coords = [t, b, l, r]

        return coords

    def set_rect_coords(self, t=None, b=None, l=None, r=None):
        """
        Called by view when dragging the part of the rectangle.
        """
        if t is not None:
            self._var_rect_top.set(t)

        if b is not None:
            self._var_rect_bottom.set(b)

        if l is not None:
            self._var_rect_left.set(l)

        if r is not None:
            self._var_rect_right.set(r)

        self.handle_rect_coord_change(redraw=False)

    def get_dashed_coord(self, canvas=True):
        coord = self.calibration.dashed_coord

        if canvas is True:
            coord = self.canvas_coords(x=coord)[0]

        return coord

    def set_dashed_coord(self, c):
        self._var_dashed.set(c)
        self.handle_dashed_coord_change(redraw=False)

    def guess_black_levels(self):
        if self.img is None:
            return

        raw = self.img.raw_image
        col = self.img.raw_colors
        t, b, l, r = self.calibration.rectangle_coords

        mask = np.zeros_like(raw, dtype=bool)
        mask[:t] = True
        mask[b:] = True
        mask[:, :l] = True
        mask[:, r:] = True

        bls = list()

        for c in range(3):
            mask_c = np.logical_and(mask, col==c)
            v = np.mean(raw[mask_c])  # Todo: check if mean is the best thing to use
            v = int(round(v))
            bls.append(v)

        self._var_global_bl_r.set(bls[0])
        self._var_global_bl_g.set(bls[1])
        self._var_global_bl_b.set(bls[2])

        self.handle_global_black_levels_change()

    def _guess_black_levels(self):
        import helper_blacklevel

        idir = None if self.img is None else os.path.dirname(self.img.fpath)
        fpaths = tk.filedialog.askopenfilename(initialdir=idir, multiple=True)

        if len(fpaths) == 0:
            return

        bls = helper_blacklevel.estimate_black_level(
            fpaths, rect=self.calibration.rectangle_coords)

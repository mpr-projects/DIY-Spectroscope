import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

from event_handler import EventHandler
from helper_misc import px


class ViewPicture:
    """
    Responsible for showing a loaded image and the associated gui to the user.
    """

    def __init__(self, controller, frame):
        self.c = controller
        self.f = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # set up canvas, putting it into it's own frame so it's easier to get
        # the size of the space available to the canvas
        self._f_canvas = f = ttk.Frame(frame)
        self._f_canvas.grid(row=0, column=0, sticky='nesw')

        self._canvas = canvas = tk.Canvas(f)  # , bg='red')
        canvas.grid(row=0, column=0, sticky='nesw')
        self._populate_canvas()

        # set up information bar at bottom
        self.ibar = ttk.Frame(frame)
        self.ibar.grid(row=1, column=0, sticky='nesw')
        self._populate_ibar()

        # set up side bar
        self.sbar = ttk.Frame(frame)
        self.sbar.grid(row=0, rowspan=2, column=1,
                       sticky='nesw', padx=10, pady=10)
        self._populate_sbar()

        self._set_up_bindings()

    def _set_up_bindings(self):
        c, canvas = self.c, self._canvas

        canvas.bind('<Motion>', c.update_mouse_info)

        for idx in [x for xs in self._map.values() for x in xs]:
            canvas.tag_bind(idx, "<ButtonPress-1>", self._canvas_click)
            canvas.tag_bind(idx, "<B1-Motion>", self._canvas_drag)
            canvas.tag_bind(idx, '<ButtonRelease-1>', self._canvas_release)

    @property
    def canvas(self):
        return self._canvas

    def _populate_canvas(self):
        canvas = self._canvas

        self._image_id = canvas.create_image(0, 0)
        self._view = None

        self._t = canvas.create_line([0, 0, 0, 0], fill='white')
        self._th = canvas.create_line(
            [0, 0, 0, 0], width=10, stipple='@transparent.xbm')

        self._b = canvas.create_line([0, 0, 0, 0], fill='white')
        self._bh = canvas.create_line(
            [0, 0, 0, 0], width=10, stipple='@transparent.xbm')

        self._l = canvas.create_line([0, 0, 0, 0], fill='white')
        self._lh = canvas.create_line(
            [0, 0, 0, 0], width=10, stipple='@transparent.xbm')

        self._r = canvas.create_line([0, 0, 0, 0], fill='white')
        self._rh = canvas.create_line(
            [0, 0, 0, 0], width=10, stipple='@transparent.xbm')

        self._v = canvas.create_line([0, 0, 0, 0], fill='white', dash=(5,1))
        self._vh = canvas.create_line(
            [0, 0, 0, 0], width=10, stipple='@transparent.xbm')

        self._map = dict(
            top=[self._t, self._th],
            bottom=[self._b, self._bh],
            vertical=[self._v, self._vh],
            left=[], right=[])

        if not self.c._restrict:
            self._map['left'] = [self._l, self._lh]
            self._map['right'] = [self._r, self._rh]

    def _populate_sbar(self):
        sbar = self.sbar
        c = self.c

        # General Section
        # --------------------------------------------------------------------
        ttk.Label(sbar, text='General').grid(
            row=0, column=0, columnspan=2, padx=10, pady=(0, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=1, column=0, columnspan=2, sticky='nesw')

        # Rectangle Coordinates
        ttk.Label(sbar, text='Rectangle Coordinates').grid(
            row=10, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5))

        ttk.Label(sbar, text='Top').grid(
            row=11, column=0, sticky='w', padx=(20, 0))

        ttk.Label(sbar, text='Bottom').grid(
            row=12, column=0, sticky='w', padx=(20, 0))

        ttk.Label(sbar, text='Left').grid(
            row=13, column=0, sticky='w', padx=(20, 0))

        ttk.Label(sbar, text='Right').grid(
            row=14, column=0, sticky='w', padx=(20, 0))

        ttk.Label(sbar, text='Dashed').grid(
            row=15, column=0, sticky='w', padx=(20, 0))

        rect_changed = self.c.handle_rect_coord_change
        dashed_changed = self.c.handle_dashed_coord_change

        t = ttk.Entry(sbar, textvariable=c._var_rect_top, width=6)
        t.grid(row=11, column=1, sticky='w')
        t.bind('<FocusOut>', rect_changed)
        t.bind('<Return>', rect_changed)

        t = ttk.Entry(sbar, textvariable=c._var_rect_bottom, width=6)
        t.grid(row=12, column=1, sticky='w')
        t.bind('<FocusOut>', rect_changed)
        t.bind('<Return>', rect_changed)

        t = ttk.Entry(sbar, textvariable=c._var_rect_left, width=6)
        t.grid(row=13, column=1, sticky='w')
        t.bind('<FocusOut>', rect_changed)
        t.bind('<Return>', rect_changed)

        if self.c._restrict:
            t.state(['disabled'])

        t = ttk.Entry(sbar, textvariable=c._var_rect_right, width=6)
        t.grid(row=14, column=1, sticky='w')
        t.bind('<FocusOut>', rect_changed)
        t.bind('<Return>', rect_changed)

        if self.c._restrict:
            t.state(['disabled'])

        t = ttk.Entry(sbar, textvariable=c._var_dashed, width=6)
        t.grid(row=15, column=1, sticky='w')
        t.bind('<FocusOut>', dashed_changed)
        t.bind('<Return>', dashed_changed)

        # Global Black Levels
        ttk.Label(sbar, text='Black Levels').grid(
            row=20, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5))

        global_bl_changed = self.c.handle_global_black_levels_change

        ttk.Label(sbar, text='Red').grid(
            row=21, column=0, sticky='w', padx=(20, 0))

        e = ttk.Entry(sbar, width=5, textvariable=c._var_global_bl_r)
        e.grid(row=21, column=1, sticky='w')
        e.bind('<FocusOut>', global_bl_changed)
        e.bind('<Return>', global_bl_changed)

        ttk.Label(sbar, text='Green').grid(
            row=22, column=0, sticky='w', padx=(20, 0))

        e = ttk.Entry(sbar, width=5, textvariable=c._var_global_bl_g)
        e.grid(row=22, column=1, sticky='w')
        e.bind('<FocusOut>', global_bl_changed)
        e.bind('<Return>', global_bl_changed)

        ttk.Label(sbar, text='Blue').grid(
            row=23, column=0, sticky='w', padx=(20, 0))

        e = ttk.Entry(sbar, width=5, textvariable=c._var_global_bl_b)
        e.grid(row=23, column=1, sticky='w')
        e.bind('<FocusOut>', global_bl_changed)
        e.bind('<Return>', global_bl_changed)

        self._btn_guess_global_bls = b = ttk.Button(
            sbar, text='Guess', command=c.guess_black_levels)
        b.grid(row=24, column=0, columnspan=2, padx=10, pady=(10, 0))

        # Lens Corrections
        ttk.Label(sbar, text='Lens Corrections').grid(
            row=30, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5))

        def lc_changed(*args):
            if self.c.have_lens_corrections_changed() is True:
                state = '!disabled'
            else:
                state = 'disabled'
            self._btn_lens_correction.state([state])

        cb = ttk.Checkbutton(
            sbar, text='Vignetting', variable=c._var_vignetting,
            command=lc_changed)
        cb.grid(row=31, column=0, columnspan=2, sticky='w', padx=20)

        if self.c._restrict:
            cb.state(['disabled'])

        cb = ttk.Checkbutton(
            sbar, text='TCA', variable=c._var_tca,
            command=lc_changed)
        cb.grid(row=32, column=0, columnspan=2, sticky='w', padx=20)

        if self.c._restrict:
            cb.state(['disabled'])

        cb = ttk.Checkbutton(
            sbar, text='Distortion', variable=c._var_distortion,
            command=lc_changed)
        cb.grid(row=33, column=0, columnspan=2, sticky='w', padx=20)

        if self.c._restrict:
            cb.state(['disabled'])

        if not self.c._restrict:
            self._btn_lens_correction = b = ttk.Button(
                sbar, text='Update', command=self.lens_correction_update)

            b.grid(row=34, column=0, columnspan=2, padx=10, pady=(5, 0))
            b.state(['disabled'])

        # Section for Picture Tab
        # --------------------------------------------------------------------
        ttk.Label(sbar, text='This Tab').grid(
            row=40, column=0, columnspan=2, padx=10, pady=(20, 5))

        ttk.Separator(sbar, orient='horizontal').grid(
            row=41, column=0, columnspan=2, sticky='nesw')

        # Show Colors
        ttk.Label(sbar, text='Show Channels').grid(
            row=50, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5))

        def show_changed(*args):
            c.handle_show_channel_change()
            # Todo: update alpha

        ttk.Checkbutton(
            sbar, text='Red', variable=c._var_show_r,
            command=show_changed).grid(
                row=51, column=0, columnspan=2, sticky='w', padx=20)

        ttk.Checkbutton(
            sbar, text='Green', variable=c._var_show_g,
            command=show_changed).grid(
                row=52, column=0, columnspan=2, sticky='w', padx=20)

        ttk.Checkbutton(
            sbar, text='Blue', variable=c._var_show_b,
            command=show_changed).grid(
                row=53, column=0, columnspan=2, sticky='w', padx=20)

        # Black Levels for Picture Tab
        ttk.Label(sbar, text='Black Level').grid(
            row=60, column=0, columnspan=2, sticky='w', padx=10, pady=(15, 5))

        bl_changed = c.handle_local_black_levels_change

        ttk.Label(sbar, text='Red').grid(
            row=61, column=0, sticky='w', padx=(20, 0))
        t = ttk.Spinbox(
            sbar, textvariable=c._var_bl_r, width=5, increment=100)
        t.grid(row=61, column=1, sticky='w')
        t.bind('<FocusOut>', bl_changed)
        t.bind('<Return>', bl_changed)
        t.bind('<<Increment>>', bl_changed)
        t.bind('<<Decrement>>', bl_changed)

        ttk.Label(sbar, text='Green').grid(
            row=62, column=0, sticky='w', padx=(20, 0))
        t = ttk.Spinbox(
            sbar, textvariable=c._var_bl_g, width=5, increment=100)
        t.grid(row=62, column=1, sticky='w')
        t.bind('<FocusOut>', bl_changed)
        t.bind('<Return>', bl_changed)
        t.bind('<<Increment>>', bl_changed)
        t.bind('<<Decrement>>', bl_changed)

        ttk.Label(sbar, text='Blue').grid(
            row=63, column=0, sticky='w', padx=(20, 0))
        t = ttk.Spinbox(
            sbar, textvariable=c._var_bl_b, width=5, increment=100)
        t.grid(row=63, column=1, sticky='w')
        t.bind('<FocusOut>', bl_changed)
        t.bind('<Return>', bl_changed)
        t.bind('<<Increment>>', bl_changed)
        t.bind('<<Decrement>>', bl_changed)

        # button to save image
        b = ttk.Button(sbar, text='Save Image', command=c.save_image_as)
        b.grid(row=70, column=0, columnspan=2, padx=10, pady=15)

    def _populate_ibar(self):
        # show information about photosite under current mouse position
        l = ttk.Label(self.ibar, textvariable=self.c._var_mouse_info)
        l.grid(row=0, column=0, padx=5, pady=5)

        # self._canvas.bind('<Motion>', self._update_mouse_info)

    def lens_correction_update(self, *args):
        self.c.handle_lens_correction_change()
        self._btn_lens_correction.state(['disabled'])

    def get_canvas_size(self):
        return self._f_canvas.winfo_width(), self._f_canvas.winfo_height()

    def resize(self, width, height):
        self._canvas.config(width=width, height=height)
        # EventHandler.generate(f'<<Canvas_Changed>>')

    def draw(self):
        if self.c.img is None:
            return

        self.draw_picture()
        self.draw_rect()

    def draw_picture(self, *args):
        c = self.c
        canvas = self._canvas

        wc, hc = px(c.canvas_width, c.canvas_height)
        view = px(*c._view_img)

        img = c._img.resize((wc, hc), box=view)
        self._imgtk = img = ImageTk.PhotoImage(image=img)  # need to store a reference to the image
        x, y = px(wc/2, hc/2)

        canvas.itemconfigure(self._image_id, image=img)
        canvas.coords(self._image_id, x, y)
        canvas.lower(self._image_id)

    def draw_rect(self):
        if self.c.img is None:
            return

        c = self.c

        t, b, l, r = c.get_rect_coords(canvas=True)
        dc = c.get_dashed_coord(canvas=True)

        self._canvas.coords(self._t, l, t, r, t)
        self._canvas.coords(self._th, l, t, r, t)

        self._canvas.coords(self._b, l, b, r, b)
        self._canvas.coords(self._bh, l, b, r, b)

        self._canvas.coords(self._l, l, t, l, b)
        self._canvas.coords(self._lh, l, t, l, b)

        self._canvas.coords(self._r, r, t, r, b)
        self._canvas.coords(self._rh, r, t, r, b)

        self._canvas.coords(self._v, dc, t, dc, b)
        self._canvas.coords(self._vh, dc, t, dc, b)

    def _canvas_click(self, event):
        c, canvas = self.c, self._canvas

        mx = canvas.canvasx(event.x)
        my = canvas.canvasy(event.y)

        ix, iy = c.image_coords(mx, my)
        self._clicked_image_x, self._clicked_image_y = ix, iy

        self._drag_id = event.widget.find_closest(event.x, event.y)[0]

    def _constrain_position(self, t, b, l, r, dc):
        wi, hi = self.c._img.size

        t = min(hi-10, max(t, 10))
        b = min(hi-10, max(b, 10))
        l = min(wi-10, max(l, 10))
        r = min(wi-10, max(r, 10))

        if self._drag_id in self._map['top']:
            t = min(t, b-2)

        if self._drag_id in self._map['left']:
            l = min(l, r-2)

        if self._drag_id in self._map['bottom']:
            b = max(b, t+2)

        if self._drag_id in self._map['right']:
            r = max(r, l+2)

        dc = max(l+1, min(r-1, dc))
        return px(t, b, l, r, dc)

    def _canvas_drag(self, event):
        c, canvas = self.c, self._canvas
        t, b, l, r = c.calibration.rectangle_coords
        dc = c.calibration.dashed_coord

        mx = canvas.canvasx(event.x)
        my = canvas.canvasy(event.y)

        ix, iy = c.image_coords(mx, my)
        dx, dy = self._clicked_image_x - ix, self._clicked_image_y - iy
        self._clicked_image_x, self._clicked_image_y = ix, iy

        if self._drag_id in self._map['top']:
            t -= dy

        if self._drag_id in self._map['bottom']:
            b -= dy

        if self._drag_id in self._map['left']:
            l -= dx

        if self._drag_id in self._map['right']:
            r -= dx

        if self._drag_id in self._map['vertical']:
            dc -= dx

        t, b, l, r, dc = self._constrain_position(t, b, l, r, dc)
        c.set_rect_coords(t=t, b=b, l=l, r=r)
        c.set_dashed_coord(dc)
        self.draw_rect()

    def _canvas_release(self, event):
        l = self._map['top'] + self._map['bottom'] \
            + self._map['left'] + self._map['right']

        if self._drag_id in l:
            EventHandler.generate('<<RectChanged>>')

import os
import tkinter as tk
import numpy as np

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_response_mixing import ViewResponseMixing


class ControllerResponseMixing(ControllerHelper):

    _cm_vars_  = [  # channel mixing
        '_var_fade_in_start_r',
        '_var_fade_in_end_r',
        '_var_fade_out_start_r',
        '_var_fade_out_end_r',
        '_var_fade_in_start_g',
        '_var_fade_in_end_g',
        '_var_fade_out_start_g',
        '_var_fade_out_end_g',
        '_var_fade_in_start_b',
        '_var_fade_in_end_b',
        '_var_fade_out_start_b',
        '_var_fade_out_end_b',
    ]

    def __init__(self, controller, frame):
        super().__init__(controller, frame)

        self._set_up_variables()
        self._view = ViewResponseMixing(self, frame)
        self._set_up_bindings()

    def _set_up_variables(self):
        for v in self._cm_vars_:
            setattr(self, v, tk.StringVar())

        self._var_plot_combined = tk.BooleanVar(value=True)
        self._var_plot_boundaries = tk.BooleanVar(value=True)

        self._var_d_name = tk.StringVar()
        self._var_d_name.trace_add(
            'write', self.handle_response_function_change)

        self._var_d_plot = tk.BooleanVar(value=False)
        self._var_d_use = tk.BooleanVar(value=False)
        self._var_d_scale = tk.DoubleVar(value=1)

        for v in self._cm_vars_:
            v = v[:4] + '_d' + v[4:]
            setattr(self, v, tk.StringVar())

    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)
        EventHandler.bind(
            '<<ControllerLoadedState>>', self._loaded_state_callback)

    def _reload_variables(self):
        cm = self.calibration.channel_mixing

        for idx, v in enumerate(self._cm_vars_):
            var = getattr(self, v, '')
            var.set(cm[idx])

        rf_names = self.calibration.response_functions_names
        self._view.set_response_functions_options(rf_names)

        name = self._var_d_name.get()

        if (name == '' or name not in rf_names) and len(rf_names) > 0:
            self._var_d_name.set(rf_names[0])

    def _clicked_tab(self, params):
        if params.clicked_name != 'Response Mixing':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._reload_variables()
        self._view.update_plot()

    def _loaded_state_callback(self, *args):
        self._reload_variables()
        self._view.update_plot()

    def handle_channel_mixing_change(self, *args):
        cm = [getattr(self, v).get() for v in self._cm_vars_]

        if cm == self.calibration.channel_mixing:
            return

        self._c.has_changed = True
        self.calibration.channel_mixing = cm

        self.save_combined_response_function_fading(cm=cm)
        self._view.update_plot()

    def save_combined_response_function_fading(self, cm=None):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        if cm is None:
            cm = [getattr(self, v).get() for v in self._cm_vars_]

        # if there's just one used rf then we're overwriting the detailed
        # fade-values with the global ones, shouldn't normally be an issue
        # (as long as the user check if the combined rf looks correct)
        for var, val in zip(self._cm_vars_, cm):
            var = var[5:]
            setattr(crf, var, val)

    def handle_response_function_change(self, *args):
        name = self._var_d_name.get()
        rf = self.calibration.get_response_function(name)

        self._var_d_plot.set(rf.plot)
        self._var_d_use.set(rf.use)
        self._var_d_scale.set(rf.scale)
        
        for c in ['r', 'g', 'b']:
            for io in ['in', 'out']:
                for se in ['start', 'end']:
                    var = f'fade_{io}_{se}_{c}'
                    val = getattr(rf, var)
                    var = f'_var_d_fade_{io}_{se}_{c}'
                    getattr(self, var).set(val)

    def handle_function_details_change(self, *args):
        name = self._var_d_name.get()
        rf = self.calibration.get_response_function(name)

        for c in ['r', 'g', 'b']:
            for io in ['in', 'out']:
                for se in ['start', 'end']:
                    var = f'_var_d_fade_{io}_{se}_{c}'
                    val = getattr(self, var).get()
                    var = f'fade_{io}_{se}_{c}'

                    if val != getattr(rf, var):
                        self._c.has_changed = True

                    setattr(rf, var, val)

        self._compute_combined_response_function()
        self._view.update_plot()

    def handle_function_details_change_plot(self, *args):
        name = self._var_d_name.get()
        rf = self.calibration.get_response_function(name)
        rf.plot = self._var_d_plot.get()
        self._view.update_plot()
        self._c.has_changed = True

    def handle_function_details_change_use(self, *args):
        name = self._var_d_name.get()
        rf = self.calibration.get_response_function(name)
        rf.use = self._var_d_use.get()
        self._c.has_changed = True

        self._compute_combined_response_function()
        self._view.update_plot()

    def handle_function_details_change_scale(self, *args):
        name = self._var_d_name.get()
        rf = self.calibration.get_response_function(name)
        val = self._var_d_scale.get()

        if val == rf.scale:
            return

        rf.scale = val
        self._c.has_changed = True

        self._compute_combined_response_function()
        self._view.update_plot()

    def delete_response_function(self, *args):
        if len(self.calibration.response_functions) < 2:
            return

        name = self._var_d_name.get()
        self.calibration.delete_response_function(name)

        self._compute_combined_response_function()
        self._reload_variables()
        self._view.update_plot()

    def import_response_function(self, *args):
        ftypes = [('Binary', '*.npy'), ('Text File', '*.txt')]
        fname = tk.filedialog.askopenfilename(filetypes=ftypes)

        if len(fname) == 0:
            return

        load_fn = np.loadtxt if fname[-3:] == 'txt' else np.load
        rf = load_fn(fname)

        nms, r, g, b = rf[0], rf[1], rf[2], rf[3]
        print('nms:', nms)

        cm = [getattr(self, v).get() for v in self._cm_vars_]
        print('cm:', cm)
        has_cm = any([v != '' for v in cm])
        use_cm = 'no'

        if has_cm is True:
            use_cm = tk.messagebox.askquestion(
                title='Fading Data',
                message=('Shall I use the fade-in/-out data'
                         ' from \'Channel Mixing\'?'))
            print('use_cm:', use_cm)

        kwargs = dict()

        if use_cm == 'yes':
            for var, val in zip(self._cm_vars_, cm):
                if val == '':
                    continue
                kwargs[var[5:]] = val

        print('kwargs:', kwargs)

        self.calibration.add_response_function(
            name = os.path.basename(fname),
            pxs=None,
            nms=nms,
            r=r, g=g, b=b,
            **kwargs)

        # self._view.update_plot()
        self._reload_variables()



    def _compute_combined_response_function(self):
        rfs = self.calibration.response_functions
        rfs = [rf for rf in rfs if rf.use == True]

        if len(rfs) == 0:
            self.calibration.clear_combined_response_function()
            return

        if len(rfs) == 1:
            self.calibration.set_combined_response_function(rf=rfs[0])
            self.save_combined_response_function_fading()
            return

        pxs = rfs[0].pxs
        nms = rfs[0].nms

        vr, wr = np.zeros_like(nms), np.zeros_like(nms)
        vg, wg = np.zeros_like(nms), np.zeros_like(nms)
        vb, wb = np.zeros_like(nms), np.zeros_like(nms)

        for rf in rfs:
            r, g, b = rf.r, rf.g, rf.b
            w = rf.get_weights()

            if not rf.nms is nms and not np.allclose(rf.nms, nms):  # Todo: I think there's an error here
                # interpolate (linearly) to common wavelength
                r = np.interp(nms, rf.nms, r)
                g = np.interp(nms, rf.nms, g)
                b = np.interp(nms, rf.nms, b)
                w = [np.interp(nms, rf.nms, v) for v in w]

            vr, vg, vb = vr+r, vg+g, vb+b
            wr, wg, wb = wr+w[0], wg+w[1], wb+w[2]

        mask = np.logical_or(np.isclose(vr, 0), np.isclose(wr, 0))
        wr[mask] = 1
        vr = vr / wr
        vr[mask] = 0

        mask = np.logical_or(np.isclose(vg, 0), np.isclose(wg, 0))
        wg[mask] = 1
        vg = vg / wg
        vg[mask] = 0

        mask = np.logical_or(np.isclose(vb, 0), np.isclose(wb, 0))
        wb[mask] = 1
        vb = vb / wb
        vb[mask] = 0

        self.calibration.set_combined_response_function(
            pxs=pxs, nms=nms, r=vr, g=vg, b=vb)
        self.save_combined_response_function_fading()

    def export_crf(self):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        ftypes = [('Binary', '*.npy'), ('Text File', '*.txt')]
        fname = tk.filedialog.asksaveasfilename(filetypes=ftypes)

        if len(fname) == 0:
            return

        crf = np.stack((crf.nms, crf.r, crf.g, crf.b), axis=0)
        save_fn = np.savetxt if fname[-3:] == 'txt' else np.save
        save_fn(fname, crf)

    def export_weights(self):
        crf = self.calibration.combined_response_function

        if crf is None:
            return

        ftypes = [('Binary', '*.npy'), ('Text File', '*.txt')]
        fname = tk.filedialog.asksaveasfilename(filetypes=ftypes)

        if len(fname) == 0:
            return

        weights = crf.get_weights()
        vals = np.stack([crf.nms,] + weights, axis=0)
        save_fn = np.savetxt if fname[-3:] == 'txt' else np.save
        save_fn(fname, vals)

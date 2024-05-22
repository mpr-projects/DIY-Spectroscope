import tkinter as tk

from event_handler import EventHandler
from controller_helper import ControllerHelper
from view_wavelength_interpolation import ViewWavelengthInterpolation


class ControllerWavelengthInterpolation(ControllerHelper):

    def __init__(self, controller, frame):
        super().__init__(controller, frame)

        self._set_up_variables()
        self._view = ViewWavelengthInterpolation(self, frame)
        self._set_up_bindings()

    def _set_up_variables(self):
        self._var_interp_method = tk.StringVar()

    def _set_up_bindings(self):
        EventHandler.bind('<<Clicked_Tab>>', self._clicked_tab)
        EventHandler.bind(
            '<<ControllerLoadedState>>', self._reload_variables)

    def _reload_variables(self, *args):
        method = self.calibration.interpolation_method
        self._var_interp_method.set(method)

    def _clicked_tab(self, params):
        if params.clicked_name != 'Wavelength Interpolation':
            return

        if params.clicked_idx == params.active_idx:
            return  # already active, nothing changed

        self._reload_variables()
        self._view.update_plot()

    def handle_method_change(self, *args):
        m = self._var_interp_method.get()

        if m == self.calibration.interpolation_method:
            return

        self._c.has_changed = True
        self.calibration.interpolation_method = m
        self._view.update_plot()

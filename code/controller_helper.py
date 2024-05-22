


class ControllerHelper:
    """
    This is the parent class of the controllers of all views, except for the
    main view. It provides convenience properties and passthrough functions 
    to functions of the main controller.
    """

    def __init__(self, controller, frame):
        self._c = controller
        self._f = frame

    @property
    def img(self):
        return getattr(self._c, '_img', None)

    @property
    def calibration(self):
        return self._c.calibration

    def get_average_measured_values(self):
        return self._c.get_average_measured_values()

    def get_color_statistics(self):
        return self._c.get_color_statistics()

    def get_valid_mappings(self):
        return self._c.get_valid_mappings()

    def get_wavelengths(self, *args, **kwargs):
        return self._c.get_wavelengths(*args, **kwargs)

import copy
import functools

from model_response_function import ResponseFunction
import helper_interpolation


class Calibration:
    """
    This class holds all data relevant for calibrating the spectroscope.
    """

    _names_ = [
        'rectangle_coords',
        'dashed_coord',
        'black_levels',
        'lens_corrections',
        'tab_picture',
        'wavelength_pixel_mapping',
        'channel_mixing',
    ]

    def __init__(self):
        self.rectangle_coords = [0, 0, 0, 0]
        self.dashed_coord = 0
        self.black_levels = [0, 0, 0]
        self.lens_corrections = [False, False, False]

        self.tab_picture = dict(
            show_channels=[True, True, True],
            black_levels=[0, 0, 0]
        )

        # using easily visible spectral lines as default
        self.wavelength_pixel_mapping = [
            [430.78, 0],
            [438.355, 0],
            [486.134, 0],
            [518.362, 0],
            [527.039, 0],
            [589.29, 0],
            [656.281, 0]]

        self.interpolation_method = \
            list(helper_interpolation.mapping.keys())[0]

        self.channel_mixing = [''] * 12
        self.response_functions = []
        self._combined_response_function = None

    @property
    def response_functions_names(self):
        return [f.name for f in self.response_functions]

    def add_response_function(self, name, pxs, nms, r, g, b, **kwargs):
        self.response_functions.append(
            ResponseFunction(name, pxs, nms, r, g, b, **kwargs))

    def get_response_function(self, name):
        for idx, rf in enumerate(self.response_functions):
            if rf.name == name:
                return rf

        return None

    def delete_response_function(self, name):
        for idx, rf in enumerate(self.response_functions):
            if rf.name == name:
                break

        del self.response_functions[idx]

    @property
    def combined_response_function(self):
        if self._combined_response_function is not None:
            return self._combined_response_function

        # if there's only one response function then it's not saved a second time
        for rf in self.response_functions:
            if rf.use is True:
                return rf

        return None

    def set_combined_response_function(self, pxs=None, nms=None, r=None, g=None, b=None, rf=None):
        if rf is not None:
            self._combined_response_function = copy.deepcopy(rf)
            return

        self._combined_response_function = ResponseFunction(
            'CombinedResponseFunction', pxs, nms, r, g, b)

    def clear_combined_response_function(self):
        self._combined_response_function = None

    def load_from_dict(self, data):
        for n in self._names_:
            setattr(self, n, data[n])

        self.response_functions = \
            [ResponseFunction(**d) for d in data['response_functions']]

        if '_combined_response_function' in data:
            self._combined_response_function = ResponseFunction(
                **data['_combined_response_function'])
        else:
            self._combined_response_function = None

    def save_to_dict(self):
        d = dict()

        for n in self._names_:
            d[n] = getattr(self, n)

        d['response_functions'] = \
            [f.save_to_dict() for f in self.response_functions]

        if self._combined_response_function is not None:
            d['_combined_response_function'] = \
                self._combined_response_function.save_to_dict()

        return d

    def get_valid_nm_px_mappings(self, split=False):
        m = self.wavelength_pixel_mapping
        m = [v for v in m if v[0] != 0 and v[1] != 0]

        if len(m) < 2:
            return None

        m = sorted(m)

        if split is False:
            return m

        nm = [v[0] for v in m]
        px = [v[1] for v in m]

        return nm, px

    @property
    def has_valid_mapping(self):
        m = self.get_valid_nm_px_mappings()

        if m is None:
            return False

        return True

    def map_px_to_nm(self, target_px):
        m = self.get_valid_nm_px_mappings(split=True)

        if m is None:
            return None

        nm, px = m

        fn = helper_interpolation.mapping[self.interpolation_method]
        nms = fn(px, nm, target_px)

        return nms

    def update_response_functions_px_nm_mapping(self):
        m = self.get_valid_nm_px_mappings(split=True)

        if m is None:
            return None

        nm, px = m

        fn = helper_interpolation.mapping[self.interpolation_method]
        fn = functools.partial(fn, px, nm)

        for rf in self.response_functions:
            rf.update_px_nm_mapping(fn)

        if self._combined_response_function is not None:
            self._combined_response_function.update_px_nm_mapping(fn)


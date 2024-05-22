import numpy as np


class ResponseFunction:

    def __init__(self, name, pxs, nms, r, g, b, **kwargs):
        self.name = name
        self.pxs = np.array(pxs)
        self.nms = np.array(nms)
        self.r = np.array(r)
        self.g = np.array(g)
        self.b = np.array(b)

        self.plot = kwargs.get('plot', False)
        self.use = kwargs.get('use', False)
        self.scale = kwargs.get('scale', 1)

        for c in ['r', 'g', 'b']:
            for io in ['in', 'out']:
                for se in ['start', 'end']:
                    var = f'fade_{io}_{se}_{c}'
                    val = kwargs.get(var, '')
                    setattr(self, var, val)

    def save_to_dict(self):
        d = dict(
            name = self.name,
            pxs=self.pxs.tolist(),
            nms=self.nms.tolist(),
            r=self.r.tolist(),
            g=self.g.tolist(),
            b=self.b.tolist(),
            plot=self.plot,
            use=self.use,
            scale=self.scale
        )

        for c in ['r', 'g', 'b']:
            for io in ['in', 'out']:
                for se in ['start', 'end']:
                    var = f'fade_{io}_{se}_{c}'
                    val = getattr(self, var)
                    d[var] = val

        return d

    def update_px_nm_mapping(self, mapping_fn):
        self.nms = mapping_fn(self.pxs)

    def get_weights(self):
        """
        Returns the weights for each channel after accounting for fading. I.e.
        the return value is three arrays of the same length as the data with
        value 0, 1 or a linear interpolation between them.
        """
        weights = list()

        for c in ['r', 'g', 'b']:
            weights.append(self._get_weights_of_channel(c))

        return weights
        
    def _get_weights_of_channel(self, c):
        nms = self.nms
        fade_in = self._check_fade_values('in', c)
        fade_out = self._check_fade_values('out', c)

        if fade_in is None and fade_out is None:
            return np.ones_like(nms)

        weights = np.zeros_like(nms)
        start_main, end_main = 0, len(nms)

        if fade_in is not None:
            fade_in, f_start, f_end = fade_in

        if fade_in is False:
            # False --> cut, doesn't matter if we use start or end
            start_main = np.where(nms >= f_start)[0][0]
        
        if fade_in is True:
            idx_start = np.where(nms >= f_start)[0][0]
            idx_end = np.where(nms >= f_end)[0]
            idx_end = len(nms) if len(idx_end) == 0 else idx_end[0]

            # can only not happen when start/end are closer than nms can resolve
            if idx_end > idx_start:
                d = idx_end - 1 - idx_start
                weights[idx_start:idx_end] = np.arange(idx_end-idx_start) / d

            start_main = idx_end

        if fade_out is not None:
            fade_out, f_start, f_end = fade_out

        if fade_out is False:
            idx_end = np.where(nms >= f_end)[0]
            idx_end = len(nms) if len(idx_end) == 0 else idx_end[0]
            end_main = idx_end

        if fade_out is True:
            idx_start = np.where(nms >= f_start)[0][0]
            idx_end = np.where(nms >= f_end)[0]
            idx_end = len(nms) if len(idx_end) == 0 else idx_end[0]

            if idx_end > idx_start:
                d = idx_end - 1 - idx_start
                weights[idx_start:idx_end] = \
                    np.arange(idx_end-idx_start)[::-1] / d

            end_main = idx_start

        weights[start_main:end_main] = 1
        return weights

    def _check_fade_values(self, inOut, c):
        """
        Returns False if there is a cut, True if there is a fade, None if
        there is not data or an error in the data.
        """
        fade_start = getattr(self, f'fade_{inOut}_start_{c}')
        fade_end = getattr(self, f'fade_{inOut}_end_{c}')

        if fade_start == '' and fade_end == '':
            return None

        if fade_start == '' and fade_end != '':
            val = float(fade_end)
            return False, val, val

        if fade_start != '' and fade_end == '':
            val = float(fade_start)
            return False, val, val

        fade_start = float(fade_start)
        fade_end = float(fade_end)

        if fade_start > fade_end:
            return None

        return True, fade_start, fade_end

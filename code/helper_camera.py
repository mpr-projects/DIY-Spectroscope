import os
import rawpy
import numpy as np
import datetime
from threading import Thread
import tempfile
import subprocess


class GetAvailableCameras(Thread):
    def __init__(self, controller, cam_list):
        super().__init__()
        self._c = controller
        self._list = cam_list  # could get that from controller

    def run(self):
        if self._c._is_taking_picture is True:
            return

        output = subprocess.run(
            ['gphoto2', '--auto-detect'],
            stdout=subprocess.PIPE,
            text=True,
        )

        cameras = output.stdout.splitlines()[2:]
        cameras = [c.split('usb:') for c in cameras]
        cameras = [c[0].strip() for c in cameras if len(c) == 2]

        self._list.extend(cameras)


class KeepCameraAwake(Thread):
    def __init__(self, controller):
        super().__init__()
        self._c = controller

        self._p = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    def _cmd(self, c):
        # could also use b"..." instead of encode
        self._p.stdin.write(f'{c}\n'.encode('UTF-8'))
        self._p.stdin.flush()

    def run(self):
        if self._c._is_taking_picture is True:
            return

        cam = self._c.camera

        if cam is None:
            return

        self._cmd(f'gphoto2 --camera=\"{cam}\" --shell')
        self._cmd(f"lcd {self._c.tempdir.name}")
        self._cmd("capture-preview")

        while True:
            line = self._p.stdout.readline().decode()

            if "Error (-6: 'Unsupported operation')" in line:
                messagebox.showerror(title='Error', message=(
                    'An error ocurred when trying to capture a preview.'))
                break

            if "Saving file as" in line:
                break

        self._cmd('exit')
        # _dir.cleanup()


class GPhotoImageHelper(Thread):
    """
    This class is responsible for taking a picture from the monochrome Canon
    350D and processing it to extract the spectrogram. It also saves associated
    data such as time and a title/description provided by the user.

    This class saves a single picture. If multiple pictures are taken (e.g. to
    monitor how sunlight changes during the day), then multiple of these models
    are created and saved by the Controller.
    """

    def __init__(self, controller):
        Thread.__init__(self)
        self._c = controller

    def _cmd(self, c):
        # could also use b"..." instead of encode
        self._p.stdin.write(f'{c}\n'.encode('UTF-8'))
        self._p.stdin.flush()

    def _capture_image_and_download(self):
        self._cmd("capture-image-and-download")
        fname = None

        while True:
            line = self._p.stdout.readline().decode()
            print(line)

            if line.startswith('Saving file as'):
                fname = line[15:].strip()

            if line.startswith('Deleting file'):
                fpath = os.path.join(self._c.tempdir.name, fname)
                return fpath

            if 'Could not capture.' in line:
                messagebox.showerror(title='Error', message=(
                    'An error ocurred when trying to capture the image.'))
                return None

    def _get_shutter_speeds(self):
        # Todo: apply more efficient way of searching for ideal shutter speed
        self._cmd('get-config /main/capturesettings/shutterspeed')
        options = list()

        while True:
            line = self._p.stdout.readline().decode()

            if line.startswith('END'):
                break

            if line.startswith('Current'):
                current = line.split()[-1]
                continue

            if not line.startswith('Choice:'):
                continue

            ss = line.split()[-1]
            options.append(ss)

        return options, current

    def _find_shutter_speed(self):
        ss, current = self._get_shutter_speeds()
        idx = ss.index(current)
        is_increasing, is_decreasing = False, False

        self._set_imageformat(fmt='RAW')
        fpath_best = None

        quantile = self._c.auto_shutterspeed_quantile
        q = float(quantile) / 100

        while True:
            fpath = self._capture_image_and_download()
            f = rawpy.imread(fpath)

            max_val = np.quantile(f.raw_image, q)
            print(f'{quantile}th quantile:', max_val)

            if max_val > f.white_level * 0.9:
                idx -= 1
                print('decreasing ss to', ss[idx])
                ss_to_use = ss[idx]
                self._cmd(f'set-config /main/capturesettings/shutterspeed {ss_to_use}')

                if is_increasing is True:
                    break

                is_decreasing = True
                fpath_best = fpath

            elif max_val < f.white_level * 0.85:
                fpath_best = fpath

                if is_decreasing is True:
                    break

                is_increasing = True

                idx += 1
                print('increasing ss to', ss[idx])
                ss_to_use = ss[idx]
                self._cmd(f'set-config /main/capturesettings/shutterspeed {ss_to_use}')

            else:
                fpath_best = fpath
                break

        return fpath_best

    def _set_imageformat(self, fmt='RAW'):
        self._cmd(f'set-config /main/imgsettings/imageformat {fmt}')

    def _get_resolutions(self):
        """
        Gets available resolutions of JPG files! Not applicable to RAW.
        """
        self._cmd('get-config /main/imgsettings/imagesize')
        options = list()
        options_px = list()

        while True:
            line = self._p.stdout.readline().decode()

            if line.startswith('END'):
                break

            if not line.startswith('Choice:'):
                continue

            r = line.split()[-1]
            options.append(r)

            r = r.split('x')
            r = int(r[0]) * int(r[1])
            options_px.append(r)

        return options, options_px

    def _change_resolution(self, res='max'):
        """
        Sets resolution of JPG files! Not applicable to RAW. Parameter 'res'
        must be either a valid resolution or 'min' or 'max'.
        """
        if res in ['min', 'max']:
            rs, rs_ = self._get_resolutions()
            res_idx = np.argmin(rs_) if res == 'min' else np.argmax(rs_)
            res = rs[res_idx]
            
        self._cmd(f'set-config /main/imgsettings/imagesize {res}')

    def _acquire_image(self):
        print('opening subprocess')
        self._p = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        print('connecting to camera')
        self._cmd('gphoto2'
                  # ' --camera=\"Canon Digital Rebel XT (normal mode)\"'
                  f' --camera=\"{self._c.camera}\"'
                  ' --shell')

        print('changing to output folder')
        self._cmd(f"lcd {self._c.tempdir.name}")

        while True:
            line = self._p.stdout.readline().decode()
            print('line:', line, type(line))

            if line.startswith('*** Error: No camera found. ***'):
                messagebox.showerror(title='No Camera', message=(
                    'Couldn\'t find a camera called \"Canon Digital Rebel TX'
                    ' (normal mode)\". Make sure it\'s connected and running.'))
                return None

            # Todo: may also want to check the output if another camera is connected

            if line.startswith('Local directory now'):
                break

        if self._c.auto_shutterspeed is True:
            # automatically find shutter speed
            fpath = self._find_shutter_speed()

        else:
            print('capture image')
            fpath = self._capture_image_and_download()

        self._cmd('exit')
        self._p.terminate()
        return fpath

    def run(self):
        if self._c.camera is None:
            return  # shouldn't be possible

        fname = self._acquire_image()
        print('fname:', fname)

        if fname is None:
            return

        t = datetime.datetime.now()
        f = t.strftime('%Y%m%d_%H%M%S')

        tpe = fname.split('.')[-1]
        fname_new = f'Spectrogram_{f}.{tpe}'
        self.fpath = os.path.join(self._c.tempdir.name, fname_new)

        os.rename(fname, self.fpath)

        # make sure process has finished (so we can't connect to the
        # camera while a connection is still active)
        self._p.wait()
        self._p = None

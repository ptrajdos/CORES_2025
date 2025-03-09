from datetime import datetime
import logging
import os

def logger(logging_dir_path,log_file_name="logfile",enable_logging:bool=True):

    if enable_logging:
        date_string = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

        log_format_str = "%(asctime)s;%(levelname)s;%(message)s"
        log_date_format = "%Y-%m-%d %H:%M:%S"
        log_filename = "{}_{}.log".format(log_file_name,date_string)
        log_file_path =  os.path.join(logging_dir_path, log_filename)
        logging.basicConfig(filename=log_filename, level=logging.DEBUG, format=log_format_str, datefmt=log_date_format)
        logging.captureWarnings(True)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def errorbar(self, *args, **kwargs):
            """Override errorbar so that line is closed by default"""
            a,lines , b = super().errorbar(*args, **kwargs)
            for line in [a]:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def escape_latex(s):
    new_s = []
    for c in s:
        if c in '#$%&_{}':
            new_s.extend('\\'+c)
        elif c == '\\':
            new_s.extend('\\textbackslash{}')
        elif c == '^':
            new_s.extend('\\textasciicircum{}')
        elif c == '~':
            new_s.extend('\\textasciitilde{}')
        else:
            new_s.append(c)
    return ''.join(new_s)


from sklearn.utils import resample

class Bootstrap:
    def __init__(self, n_samples = 3, random_state = None, stratify=None) -> None:
        """
        Bootstraping procedure. For experiments only.
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y=None, groups=None):
        n_objects  = len(X)
        indices = [*range(n_objects)]

        resampled_idx = resample(indices,replace=True, n_samples= n_objects*self.n_samples, random_state=self.random_state,
                                     stratify=self.stratify)

        start = 0
        end = n_objects 
        for i in range(self.n_samples):
            train_idx = resampled_idx[start:end]
            test_idx = train_idx
            yield  train_idx, test_idx 
            start += n_objects
            end += n_objects

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_samples
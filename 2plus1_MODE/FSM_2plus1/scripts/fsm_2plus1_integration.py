"""Drop-in 2+1 complex demodulation helpers for a four-phase FSM pipeline.

Phase convention:
    I0 = A + B*cos(phi)
    I1 = A - B*sin(phi)
    I2 = A - B*cos(phi)
    I3 = A + B*sin(phi)

Complex convention:
    C = (I0 - I2) + 1j*(I3 - I1)
All inputs are CAMERA images, not projected patterns.
"""
import numpy as np


def four_step_complex(I0, I1, I2, I3):
    """Reference four-step complex response, using safe floating-point differences."""
    i0, i1, i2, i3 = [np.asarray(x, dtype=np.float64)
                       for x in (I0, I1, I2, I3)]
    return (i0 - i2) + 1j * (i3 - i1)


def two_plus_one_with_gray(I0, I1, gray_capture):
    """Use one captured uniform-gray image equal to the fringe DC baseline.

    The uniform gray illumination must match the mean PROJECTED light
    of all four shifted patterns, in linear optical intensity.
    """
    i0, i1, a = [np.asarray(x, dtype=np.float64)
                  for x in (I0, I1, gray_capture)]
    return 2.0 * ((i0 - a) + 1j * (a - i1))


def two_plus_one_with_white(I0, I1, white_capture, black_capture):
    """Use full-white and black-projection camera images.

    For linearly encoded fringes with P0=0.5*(P_white+P_black),
    and stable ambient light: A = (W + D) / 2.
    The black exposure can be reused only if ambient/settings remain stable.
    """
    w, d = [np.asarray(x, dtype=np.float64)
            for x in (white_capture, black_capture)]
    return two_plus_one_with_gray(I0, I1, (w + d) / 2.0)


def fsm_frequency_fft(complex_stack, frequency_axis=0):
    """FFT across FREQ samples; stack usually has shape (F, camera_height, camera_width).

    Keep any downstream shifts, normalizations, filtering and peak detection
    from your existing FSM code unchanged.
    """
    return np.fft.fft(np.asarray(complex_stack), axis=frequency_axis)


def relative_complex_error(estimate, reference):
    """Global normalized L2 error between complex frames/stacks."""
    a, b = np.asarray(estimate), np.asarray(reference)
    return float(np.linalg.norm(a - b) /
                 max(float(np.linalg.norm(b)), np.finfo(float).eps))

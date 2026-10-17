"""Webcam calibration and planar measurement.

Modules:
    camera_io   open a webcam, verify resolution, measure fps
    board       3D model of the checkerboard
    detect      sub-pixel corner detection, sharpness, orientation
    calibrate   camera calibration and reprojection errors
    evaluate    outlier views and hold-out validation
    geometry    intrinsics under resize/crop, pixel -> mm on a plane
    io_yaml     read/write the calibration file (OpenCV FileStorage)
"""

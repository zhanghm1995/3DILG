'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-29 11:53:16
Email: haimingzhang@link.cuhk.edu.cn
Description: The utilities for data visualization
'''


def write_obj(points, file, rgb=False):
    fout = open('%s.xyz' % file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('%f %f %f\n' % (
                points[i, 0], points[i, 1], points[i, 2]))
        else:
            fout.write('%f %f %f\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3] * 255, points[i, -2] * 255,
                points[i, -1] * 255))


def box2obj(box, objname):
    corners = box.corners().T
    with open(objname, 'w') as f:
        for corner in corners:
            f.write('v %f %f %f\n' % (corner[0], corner[1], corner[2]))
        f.write('f %d %d %d %d\n' % (1, 2, 3, 4))
        f.write('f %d %d %d %d\n' % (5, 6, 7, 8))
        f.write('f %d %d %d %d\n' % (1, 5, 8, 4))
        f.write('f %d %d %d %d\n' % (2, 6, 7, 3))
        f.write('f %d %d %d %d\n' % (1, 2, 6, 5))
        f.write('f %d %d %d %d\n' % (4, 3, 7, 8))

def save_xyz_file(numpy_array, xyz_dir):
    num_points = numpy_array.shape[0]
    with open(xyz_dir, 'w') as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (numpy_array[i, 0], numpy_array[i, 1], numpy_array[i, 2])
            f.write(line)
    return

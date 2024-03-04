from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np

for _ in range(3000):
    q1 = Quaternion.random().unit
    q2 = Quaternion.random().unit

    q_diff = q1 * q2.inverse
    q_diff_e = R.from_quat(q_diff.elements).as_euler('xyz', degrees=True)

    e1 = R.from_quat(q1.elements).as_euler('xyz', degrees=True)
    e2 = R.from_quat(q2.elements).as_euler('xyz', degrees=True)

    e_diff = e1 - e2

    print(q_diff_e == e_diff, q_diff_e, e_diff)

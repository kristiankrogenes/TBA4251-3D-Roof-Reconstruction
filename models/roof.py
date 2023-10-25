import numpy as np
from scipy.linalg import lstsq

class Roof():

    def __init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs):

        self.roof_type = roof_type
        self.roof_id = roof_id

        self.segment_ids = segment_ids

        self.segment_xs = segment_xs
        self.segment_ys = segment_ys
        self.segment_zs = segment_zs

        self.global_minz = min([min(z) for z in segment_zs])
        self.global_maxz = max([max(z) for z in segment_zs])

        self.segment_planes = self.get_segment_planes()

    def get_segment_planes(self):
        segment_planes = []
        for x, y, z in zip(self.segment_xs, self.segment_ys, self.segment_zs):
            A = np.column_stack((x, y, np.ones(len(x))))
            b = z
            coeffs, _, _, _ = lstsq(A, b)
            coeffs = [coeffs[0], coeffs[1], -1, coeffs[2]]
            segment_planes.append(coeffs)
        return segment_planes
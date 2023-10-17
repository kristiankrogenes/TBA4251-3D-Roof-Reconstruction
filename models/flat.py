from models.roof import Roof
from helpers import find_min_max_values, intersect_3planes
from shapely import MultiPoint
import numpy as np

class FlatRoof(Roof):

    def __init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs):

        Roof.__init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs)

        self.roof_polygons = self.create_roof_polygons()

    def create_roof_polygons(self):
        dv1 = self.segment_planes[0][:3]
        dv2 = np.cross(dv1, [0, 0, 1])

        x, y, z = self.segment_xs[0], self.segment_ys[0], self.segment_zs[0]
        p1, p2, p3, p4, _, _ = find_min_max_values(x, y, z)

        edge_plane_1 = [dv1[0], dv1[1], 0, -dv1[0]*p3[0]-dv1[1]*p3[1]-0*p3[2]]
        edge_plane_2 = [dv1[0], dv1[1], 0, -dv1[0]*p4[0]-dv1[1]*p4[1]-0*p4[2]]
        edge_plane_3 = [dv2[0], dv2[1], 0, -dv2[0]*p1[0]-dv2[1]*p1[1]-0*p1[2]]
        edge_plane_4 = [dv2[0], dv2[1], 0, -dv2[0]*p2[0]-dv2[1]*p2[1]-0*p2[2]]

        roof_plane = self.segment_planes[0]

        tip1, _, _ = intersect_3planes(roof_plane, edge_plane_1, edge_plane_3)
        tip2, _, _ = intersect_3planes(roof_plane, edge_plane_1, edge_plane_4)
        tip3, _, _ = intersect_3planes(roof_plane, edge_plane_2, edge_plane_3)
        tip4, _, _ = intersect_3planes(roof_plane, edge_plane_2, edge_plane_4)

        return [MultiPoint([tip1, tip2, tip3, tip4]).convex_hull]
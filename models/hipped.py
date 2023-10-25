from models.roof import Roof
from helpers import intersect_4planes, intersect_3planes, intersect_2planes, find_min_max_values, distance, check_point_sides
from shapely import MultiPoint

class HippedRoof(Roof):

    def __init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs):

        Roof.__init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs)

        self.z_min = min([min(z_coords) for z_coords in self.segment_zs])
        self.z_max = max([max(z_coords) for z_coords in self.segment_zs])

        self.gabled_segments = self.find_gabled_segments()
        self.edge_segments = self.find_edge_segments()
        
        self.roof_polygons = self.create_roof_polygons()
    
    def find_gabled_segments(self):
        gabled_matches = []
        for indexi, i in enumerate(self.segment_ids):
            for indexj, j in enumerate(self.segment_ids[indexi+1:]):
                    intersection, direction_vector = intersect_2planes(
                        self.segment_planes[indexi], 
                        self.segment_planes[indexi+1+indexj], 
                    )

                    max_height_point = intersection + self.z_max/direction_vector[2] * direction_vector
                    random_point_from_roof = [self.segment_xs[0][0], self.segment_ys[0][0], self.segment_zs[0][0]]
                    
                    if abs(direction_vector[2]) < 0.1 and distance(max_height_point, random_point_from_roof) < 500:
                        gabled_matches.append([indexi,  indexi+1+indexj])
        return gabled_matches
    
    def find_edge_segments(self):
        edge_segments = [id for id in self.segment_ids]
        for gabled_match in self.gabled_segments:
            for segment_id in gabled_match:
                edge_segments.remove(segment_id)
        return edge_segments
    
    def create_roof_polygons(self):
        roof_polygons = []

        match self.roof_type:
            case "hipped":
                roof_polygons = [None for _ in range(4)]
                tip1, eps1, dvs1 = intersect_3planes(
                    self.segment_planes[self.edge_segments[0]], 
                    self.segment_planes[self.gabled_segments[0][0]], 
                    self.segment_planes[self.gabled_segments[0][1]],
                    self.global_minz
                )
                tip2, eps2, dvs2 = intersect_3planes(
                    self.segment_planes[self.edge_segments[1]], 
                    self.segment_planes[self.gabled_segments[0][0]], 
                    self.segment_planes[self.gabled_segments[0][1]],
                    self.global_minz
                )
                roof_polygons[self.edge_segments[0]] = MultiPoint([tip1, eps1[0], eps1[1]]).convex_hull
                roof_polygons[self.edge_segments[1]] = MultiPoint([tip2, eps2[0], eps2[1]]).convex_hull
                roof_polygons[self.gabled_segments[0][0]] = MultiPoint([tip1, tip2, eps1[0], eps2[0]]).convex_hull
                roof_polygons[self.gabled_segments[0][1]] = MultiPoint([tip1, tip2, eps1[1], eps2[1]]).convex_hull
                # roof_polygons.append(MultiPoint([tip1, eps1[0], eps1[1]]).convex_hull)
                # roof_polygons.append(MultiPoint([tip2, eps2[0], eps2[1]]).convex_hull)
                # roof_polygons.append(MultiPoint([tip1, tip2, eps1[0], eps2[0]]).convex_hull)
                # roof_polygons.append(MultiPoint([tip1, tip2, eps1[1], eps2[1]]).convex_hull)
            case "corner_element":
                roof_polygons = [None for _ in range(6)]

                edge_top_points = []
                edge_bottom_points = []
                opposites = []
                left_right_segments = []
                for edge in self.edge_segments:
                    
                    for j in range(len(self.gabled_segments)):
                        tip, eps, dvs = intersect_3planes(
                            self.segment_planes[edge], 
                            self.segment_planes[self.gabled_segments[j][0]], 
                            self.segment_planes[self.gabled_segments[j][1]], 
                            self.global_minz
                        )
                        if not len(eps) == 0:
                            edge_top_points.append(tip)
                            
                            nj = 0 if j==1 else 1  

                            _, _, _, _, _, maxz0 = find_min_max_values(self.segment_xs[edge], self.segment_ys[edge], self.segment_zs[edge])
                            minx1, maxx1, _, _, _, _ = find_min_max_values(self.segment_xs[self.gabled_segments[0][0]], self.segment_ys[self.gabled_segments[0][0]], self.segment_zs[self.gabled_segments[0][0]])
                            minx2, maxx2, _, _, _, _ = find_min_max_values(self.segment_xs[self.gabled_segments[0][1]], self.segment_ys[self.gabled_segments[0][1]], self.segment_zs[self.gabled_segments[0][1]])

                            opposite_segments = [0, 1] if distance(minx1, maxz0)+distance(maxx1, maxz0) > distance(minx2, maxz0)+distance(maxx2, maxz0) else [1, 0]

                            left, right = check_point_sides(tip, eps[0], eps[1], self.gabled_segments[j][0], self.gabled_segments[j][1])
                            edge_bottom_points.append([left[1], right[1]])
                            left_right_segments.append([left[0], right[0]])
                            opposites.append([self.gabled_segments[nj][opposite_segments[0]], self.gabled_segments[nj][opposite_segments[1]]])

                            break

                intersection1, dv1 = intersect_2planes(
                    self.segment_planes[left_right_segments[0][0]], 
                    self.segment_planes[left_right_segments[1][1]],
                )
                bottom_center_point1 = intersection1 + self.global_minz/dv1[2] * dv1

                intersection2, dv2 = intersect_2planes(
                    self.segment_planes[left_right_segments[0][1]], 
                    self.segment_planes[left_right_segments[1][0]], 
                )
                bottom_center_point2 = intersection2 + self.global_minz/dv2[2] * dv2

                qip = intersect_4planes(
                    self.segment_planes[opposites[0][0]], 
                    self.segment_planes[opposites[1][0]], 
                    self.segment_planes[opposites[0][1]], 
                    self.segment_planes[opposites[1][1]], 
                )

                roof_polygons[self.edge_segments[0]] = MultiPoint([edge_top_points[0], edge_bottom_points[0][0], edge_bottom_points[0][1]]).convex_hull
                roof_polygons[self.edge_segments[1]] = MultiPoint([edge_top_points[1], edge_bottom_points[1][0], edge_bottom_points[1][1]]).convex_hull
                roof_polygons[left_right_segments[0][0]] = MultiPoint([edge_top_points[0], qip, bottom_center_point1, edge_bottom_points[0][0]]).convex_hull
                roof_polygons[left_right_segments[0][1]] = MultiPoint([edge_top_points[0], qip, bottom_center_point2, edge_bottom_points[0][1]]).convex_hull
                roof_polygons[left_right_segments[1][0]] = MultiPoint([edge_top_points[1], qip, bottom_center_point1, edge_bottom_points[1][1]]).convex_hull
                roof_polygons[left_right_segments[1][1]] = MultiPoint([edge_top_points[1], qip, bottom_center_point2, edge_bottom_points[1][0]]).convex_hull
                # roof_polygons.append(MultiPoint([edge_top_points[0], edge_bottom_points[0][0], edge_bottom_points[0][1]]).convex_hull)
                # roof_polygons.append(MultiPoint([edge_top_points[1], edge_bottom_points[1][0], edge_bottom_points[1][1]]).convex_hull)

                # roof_polygons.append(MultiPoint([edge_top_points[0], qip, bottom_center_point1, edge_bottom_points[0][0]]).convex_hull)
                # roof_polygons.append(MultiPoint([edge_top_points[0], qip, bottom_center_point2, edge_bottom_points[0][1]]).convex_hull)
                
                # roof_polygons.append(MultiPoint([edge_top_points[1], qip, bottom_center_point1, edge_bottom_points[1][1]]).convex_hull)
                # roof_polygons.append(MultiPoint([edge_top_points[1], qip, bottom_center_point2, edge_bottom_points[1][0]]).convex_hull)

        return roof_polygons
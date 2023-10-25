from models.roof import Roof
from helpers import intersect_2planes, distance, find_min_max_values, intersect_3planes
from shapely import Polygon, MultiPoint

class GabledRoof(Roof):

    def __init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs):

        Roof.__init__(self, roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs)

        self.gabled_segments = self.find_gabled_segments()
        self.main_gabled_segments = self.find_main_gabled_segments()
        self.sub_gabled_segments = self.find_sub_gabled_segments()

        self.roof_polygons = self.create_roof_polygons()

    def find_gabled_segments(self):
        if (self.segment_ids == 2):
            return [self.segment_ids]
        else:
            gabled_matches = []
            for indexi, i in enumerate(self.segment_ids):
                for indexj, j in enumerate(self.segment_ids[indexi+1:]):
                        intersection, direction_vector = intersect_2planes(
                            self.segment_planes[indexi], 
                            self.segment_planes[indexi+1+indexj], 
                        )
                        if abs(direction_vector[2]) < 0.1:
                            if self.segment_planes[indexi][3]>0 and self.segment_planes[indexi+1+indexj][3]<0 or self.segment_planes[indexi][3]<0 and self.segment_planes[indexi+1+indexj][3]>0:
                                gabled_matches.append([indexi,  indexi+1+indexj])
            return gabled_matches

    def find_main_gabled_segments(self):
        if len(self.gabled_segments) == 1:
            return self.gabled_segments[0]
        if len(self.gabled_segments) == 2:
            return self.gabled_segments[0] if max(self.segment_zs[self.gabled_segments[0][0]]) > max(self.segment_zs[self.gabled_segments[1][0]]) else self.gabled_segments[1]
        main_roof_match_index = None
        for i in self.segment_ids:
            count = []
            for j in range(len(self.gabled_segments)):
                if i in self.gabled_segments[j]:
                    count.append(j)
            if len(count) == 1:
                main_roof_match_index = count[0]
                break
        return self.gabled_segments[main_roof_match_index]

    def find_sub_gabled_segments(self):
        if self.segment_ids == 2:
            return None
        
        sub_gabled_matches = [match for match in self.gabled_segments if match[0] != self.main_gabled_segments[0] and match[1] != self.main_gabled_segments[1]]
        
        if len(sub_gabled_matches) == 1:
            return sub_gabled_matches
        
        sub_gabled_match_distances = []

        for match in sub_gabled_matches:
            points_seg1 = [[xi, yi, zi] for xi, yi, zi in zip(self.segment_xs[match[0]], self.segment_ys[match[0]], self.segment_zs[match[0]])]
            points_seg2 = [[xi, yi, zi] for xi, yi, zi in zip(self.segment_xs[match[1]], self.segment_ys[match[1]], self.segment_zs[match[1]])]
            
            match_distance = []
            for point1 in points_seg1:
                total_distance = 0
                for point2 in points_seg2:
                    total_distance += distance(point1, point2)
                match_distance.append(total_distance / len(points_seg2))
            sub_gabled_match_distances.append(sum(match_distance) / len(match_distance))
            
        correct_matches = []
        for i in range(len(sub_gabled_matches)):
            for j in range(i+1, len(sub_gabled_matches)):
                if sub_gabled_matches[i][0] == sub_gabled_matches[j][0]:
                    if sub_gabled_match_distances[i] < sub_gabled_match_distances[j]:
                        correct_matches.append(sub_gabled_matches[i])
                    else:
                        correct_matches.append(sub_gabled_matches[j])
                    break
        return correct_matches
    
    def create_roof_polygons(self):

        roof_polygons = [None for _ in range(len(self.segment_ids))]

        minx_main0, maxx_main0, _, _, minz_main0, _ = find_min_max_values(self.segment_xs[self.main_gabled_segments[0]], self.segment_ys[self.main_gabled_segments[0]], self.segment_zs[self.main_gabled_segments[0]])
        _, _, _, _, minz_main1, _ = find_min_max_values(self.segment_xs[self.main_gabled_segments[1]], self.segment_ys[self.main_gabled_segments[1]], self.segment_zs[self.main_gabled_segments[1]])
        
        _, main_intersection_dv = intersect_2planes(self.segment_planes[self.main_gabled_segments[0]], self.segment_planes[self.main_gabled_segments[1]])
        main_edge_plane1 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*minx_main0[0] - main_intersection_dv[1]*minx_main0[1] - 0*minx_main0[2]]
        main_edge_plane2 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*maxx_main0[0] - main_intersection_dv[1]*maxx_main0[1] - 0*maxx_main0[2]]

        main_tip1, main_ips1, main_dvs1 = intersect_3planes(
            self.segment_planes[self.main_gabled_segments[0]],
            self.segment_planes[self.main_gabled_segments[1]],
            main_edge_plane1,
            self.global_minz
        )

        main_tip2, main_ips2, main_dvs2 = intersect_3planes(
            self.segment_planes[self.main_gabled_segments[0]],
            self.segment_planes[self.main_gabled_segments[1]],
            main_edge_plane2,
            self.global_minz
        )

        main_points = [{"main_tip": main_tip1, "main_ips": main_ips1}, {"main_tip": main_tip2, "main_ips": main_ips2}]

        if len(self.segment_ids) == 2:
            roof_polygons[0] = MultiPoint([main_points[0]["main_tip"], main_points[1]["main_tip"], main_points[0]["main_ips"][0], main_points[1]["main_ips"][0]]).convex_hull
            roof_polygons[1] = MultiPoint([main_points[0]["main_tip"], main_points[1]["main_tip"], main_points[0]["main_ips"][1], main_points[1]["main_ips"][1]]).convex_hull
        else:       
            used_main_segment = []
            for sub_match in self.sub_gabled_segments:
                minx_sub0, maxx_sub0, _, _, minz_sub0, _ = find_min_max_values(self.segment_xs[sub_match[0]], self.segment_ys[sub_match[0]], self.segment_zs[sub_match[0]])
                minx_sub1, _, _, _, minz_sub1, _ = find_min_max_values(self.segment_xs[sub_match[1]], self.segment_ys[sub_match[1]], self.segment_zs[sub_match[1]])

                closest_main_segment_index = 0 if distance(minz_sub0, minz_main0)+distance(minz_sub1, minz_main0) < distance(minz_sub0, minz_main1)+distance(minz_sub1, minz_main1) else 1
                sub_seg_sides = [0, 1] if distance(minx_sub0, minx_main0) < distance(minx_sub1, minx_main0) else [1, 0]

                main_sub_tip, main_sub_ips, main_sub_dvs = intersect_3planes(
                    self.segment_planes[sub_match[sub_seg_sides[0]]], 
                    self.segment_planes[sub_match[sub_seg_sides[1]]], 
                    self.segment_planes[self.main_gabled_segments[closest_main_segment_index]],
                    self.global_minz
                )

                sub_edge_ref_point = minx_sub0 if distance(minx_sub0, main_sub_tip) > distance(maxx_sub0, main_sub_tip) else maxx_sub0
                sub0_edge_plane = [main_sub_dvs[0][0], main_sub_dvs[0][1], 0, -main_sub_dvs[0][0]*sub_edge_ref_point[0] - main_sub_dvs[0][1]*sub_edge_ref_point[1]]
            
                sub_tip, sub_ips, sub_dvs = intersect_3planes(
                    self.segment_planes[sub_match[sub_seg_sides[0]]], 
                    self.segment_planes[sub_match[sub_seg_sides[1]]], 
                    sub0_edge_plane,
                    self.global_minz
                )

                opposite_index = 1 if closest_main_segment_index==0 else 0

                roof_polygons[sub_match[sub_seg_sides[1]]] = MultiPoint([main_sub_tip, sub_tip, main_sub_ips[0], sub_ips[0]]).convex_hull
                roof_polygons[sub_match[sub_seg_sides[0]]] = MultiPoint([main_sub_tip, sub_tip, main_sub_ips[1], sub_ips[1]]).convex_hull
                roof_polygons[self.main_gabled_segments[closest_main_segment_index]] = Polygon([
                    main_sub_tip, 
                    main_sub_ips[1], 
                    main_points[0]["main_ips"][opposite_index],
                    main_points[0]["main_tip"],
                    main_points[1]["main_tip"],
                    main_points[1]["main_ips"][opposite_index],
                    main_sub_ips[0], 
                ])
                # roof_polygons.append(MultiPoint([main_sub_tip, sub_tip, main_sub_ips[0], sub_ips[0]]).convex_hull)
                # roof_polygons.append(MultiPoint([main_sub_tip, sub_tip, main_sub_ips[1], sub_ips[1]]).convex_hull)
                # roof_polygons.append(Polygon([
                #     main_sub_tip, 
                #     main_sub_ips[1], 
                #     main_points[0]["main_ips"][opposite_index],
                #     main_points[0]["main_tip"],
                #     main_points[1]["main_tip"],
                #     main_points[1]["main_ips"][opposite_index],
                #     main_sub_ips[0], 
                # ]))

                used_main_segment.append(closest_main_segment_index)
                
            if len(used_main_segment) == 1:
                opposite_index = 1 if used_main_segment[0]==0 else 0
                roof_polygons[self.main_gabled_segments[opposite_index]] = MultiPoint([main_points[0]["main_tip"], main_points[1]["main_tip"], main_points[0]["main_ips"][used_main_segment[0]], main_points[1]["main_ips"][used_main_segment[0]]]).convex_hull
                # roof_polygons.append(MultiPoint([main_points[0]["main_tip"], main_points[1]["main_tip"], main_points[0]["main_ips"][used_main_segment[0]], main_points[1]["main_ips"][used_main_segment[0]]]).convex_hull)
        
        return roof_polygons
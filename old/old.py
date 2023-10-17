"""
def sort_gabled_roof_segments(self, roof_type, roof_id):
    intersections = self.intersection_df.loc[self.intersection_df['roof_id'] == roof_id]
    intersections = intersections.reset_index()

    roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]

    x1, y1, z1 = list(roof["x_coordinates"])[0], list(roof["y_coordinates"])[0], list(roof["z_coordinates"])[0]
    x2, y2, z2 = list(roof["x_coordinates"])[1], list(roof["y_coordinates"])[1], list(roof["z_coordinates"])[1]

    segment_ids = list(roof["segment_id"])
    segment_planes = list(roof["plane_coefficients"])

    xs, ys, zs = list(roof["x_coordinates"]), list(roof["y_coordinates"]), list(roof["z_coordinates"])
    gr1 = GabledRoof(roof_type, roof_id, segment_ids, segment_planes, xs, ys, zs)
    return gr1
    
    s1p1, s1p2, _, _, s1p5, _ = find_min_max_values(x1, y1, z1)
    _, _, _, _, s2p5, _ = find_min_max_values(x2, y2, z2)

    labeled_points = {}

    plane1, plane2 = list(roof["plane_coefficients"])[0], list(roof["plane_coefficients"])[1]

    direction_vector = intersections.loc[0]["direction_vector"]
    
    edge_plane1 = [direction_vector[0], direction_vector[1], 0, -direction_vector[0]*s1p1[0] - direction_vector[1]*s1p1[1] - 0*s1p1[2]]
    edge_plane2 = [direction_vector[0], direction_vector[1], 0, -direction_vector[0]*s1p2[0] - direction_vector[1]*s1p2[1] - 0*s1p2[2]]

    tip1, ips1, dvs1 = intersect_3planes(plane1, plane2, edge_plane1)
    tip2, ips2, dvs2 = intersect_3planes(plane1, plane2, edge_plane2)

    EPs1p1 = ips1[2]+s1p5[2]/dvs1[2][2] * dvs1[2]
    EPs1p2 = ips2[2]+s1p5[2]/dvs2[2][2] * dvs2[2]
    EPs2p1 = ips1[1]+s2p5[2]/dvs1[1][2] * dvs1[1]
    EPs2p2 = ips2[1]+s2p5[2]/dvs2[1][2] * dvs2[1]

    labeled_points["p1"] = [tip1, tip2, EPs1p1, EPs1p2]
    labeled_points["p2"] = [tip1, tip2, EPs2p1, EPs2p2]

    return self.create_polygons(roof_type, labeled_points), gr1
    
def sort_cross_element_roof_segments(self, roof_type, roof_id):
    roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
    segment_ids = list(roof["segment_id"])
    segment_planes = list(roof["plane_coefficients"])

    xs, ys, zs = list(roof["x_coordinates"]), list(roof["y_coordinates"]), list(roof["z_coordinates"])
    intersections, gabled_matches, main_index, sub0_index, sub1_index = find_segment_matches(segment_ids, segment_planes, xs, ys, zs, True)
    main, sub0, sub1 = gabled_matches[main_index], gabled_matches[sub0_index], gabled_matches[sub1_index]

    gr1 = GabledRoof(roof_type, roof_id, segment_ids, segment_planes, xs, ys, zs)
    return gr1
    
    minx_sub0_seg0, maxx_sub0_seg0, _, _, minz_sub0_seg0, _ = find_min_max_values(xs[sub0[0]], ys[sub0[0]], zs[sub0[0]])
    minx_sub1_seg0, maxx_sub1_seg0, _, _, minz_sub1_seg0, _ = find_min_max_values(xs[sub1[0]], ys[sub1[0]], zs[sub1[0]])
    minx_main_seg0, maxx_main_seg0, _, _, minz_main_seg0, _ = find_min_max_values(xs[main[0]], ys[main[0]], zs[main[0]])
    minx_main_seg1, maxx_main_seg1, _, _, minz_main_seg1, _ = find_min_max_values(xs[main[1]], ys[main[1]], zs[main[1]])

    _, _, _, _, minz_sub0_seg1, _ = find_min_max_values(xs[sub0[1]], ys[sub0[1]], zs[sub0[1]])
    _, _, _, _, minz_sub1_seg1, _ = find_min_max_values(xs[sub1[1]], ys[sub1[1]], zs[sub1[1]])

    global_zmin = min(minz_sub0_seg0[2], minz_sub1_seg0[2], minz_main_seg0[2], minz_main_seg1[2])

    # MAIN 
    closest_main_sub0 = 0 if distance(minz_sub0_seg0, minz_main_seg0)+distance(minz_sub0_seg1, minz_main_seg0) < distance(minz_sub0_seg0, minz_main_seg1)+distance(minz_sub0_seg1, minz_main_seg1) else 1
    main_sub0_tip, main_sub0_ips, main_sub0_dvs = intersect_3planes(
        segment_planes[sub0[0]], 
        segment_planes[sub0[1]], 
        segment_planes[main[closest_main_sub0]]
    )
    main_sub0_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(main_sub0_ips, main_sub0_dvs) if abs(dv[2]) > 0.1]

    # MAIN SUB1
    closest_main_sub1 = 0 if distance(minz_sub1_seg0, minz_main_seg0)+distance(minz_sub1_seg1, minz_main_seg0) < distance(minz_sub1_seg0, minz_main_seg1)+distance(minz_sub1_seg1, minz_main_seg1) else 1
    main_sub1_tip, main_sub1_ips, main_sub1_dvs = intersect_3planes(
        segment_planes[sub1[0]], 
        segment_planes[sub1[1]], 
        segment_planes[main[closest_main_sub1]]
    )
    main_sub1_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(main_sub1_ips, main_sub1_dvs) if abs(dv[2]) > 0.1]

    # SUB0 EDGE
    sub0_edge_ref_point = minx_sub0_seg0 if distance(minx_sub0_seg0, main_sub0_tip) > distance(maxx_sub0_seg0, main_sub0_tip) else maxx_sub0_seg0
    sub0_intersection_dv = intersections[sub0_index][1]
    sub0_edge_plane = [sub0_intersection_dv[0], sub0_intersection_dv[1], 0, -sub0_intersection_dv[0]*sub0_edge_ref_point[0] - sub0_intersection_dv[1]*sub0_edge_ref_point[1] - 0*sub0_edge_ref_point[2]]
    
    sub0_tip, sub0_ips, sub0_dvs = intersect_3planes(
        segment_planes[sub0[0]], 
        segment_planes[sub0[1]], 
        sub0_edge_plane
    )
    sub0_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(sub0_ips, sub0_dvs) if abs(dv[2]) > 0.1]

    # SUB1 EDGE
    sub1_edge_ref_point = minx_sub1_seg0 if distance(minx_sub1_seg0, main_sub1_tip) > distance(maxx_sub1_seg0, main_sub1_tip) else maxx_sub1_seg0
    sub1_intersection_dv = intersections[sub1_index][1]
    sub1_edge_plane = [sub1_intersection_dv[0], sub1_intersection_dv[1], 0, -sub1_intersection_dv[0]*sub1_edge_ref_point[0] - sub1_intersection_dv[1]*sub1_edge_ref_point[1] - 0*sub1_edge_ref_point[2]]
    
    sub1_tip, sub1_ips, sub1_dvs = intersect_3planes(
        segment_planes[sub1[0]], 
        segment_planes[sub1[1]], 
        sub1_edge_plane
    )
    sub1_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(sub1_ips, sub1_dvs) if abs(dv[2]) > 0.1]

    # MAIN EDGE1
    main_intersection_dv = intersections[main_index][1]
    main_edge_plane1 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*minx_main_seg0[0] - main_intersection_dv[1]*minx_main_seg0[1] - 0*minx_main_seg0[2]]
    main_edge_plane2 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*maxx_main_seg0[0] - main_intersection_dv[1]*maxx_main_seg0[1] - 0*maxx_main_seg0[2]]

    main_tip1, main_ips1, main_dvs1 = intersect_3planes(
        segment_planes[main[0]], 
        segment_planes[main[1]], 
        main_edge_plane1
    )
    main_eps1 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips1, main_dvs1) if abs(dv[2]) > 0.1]

    main_tip2, main_ips2, main_dvs2 = intersect_3planes(
        segment_planes[main[0]], 
        segment_planes[main[1]], 
        main_edge_plane2
    )
    main_eps2 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips2, main_dvs2) if abs(dv[2]) > 0.1]

    ind0 = 1 if closest_main_sub0==0 else 0
    ind1 = 1 if closest_main_sub1==0 else 0

    return self.create_polygons(roof_type, {
        "p1": [main_sub0_tip, main_sub0_eps[0], sub0_eps[0], sub0_tip],
        "p2": [main_sub0_tip, main_sub0_eps[1], sub0_eps[1], sub0_tip],
        "p3": [main_sub1_tip, main_sub1_eps[0], sub1_eps[0], sub1_tip],
        "p4": [main_sub1_tip, main_sub1_eps[1], sub1_eps[1], sub1_tip],
        "p5": [main_sub0_tip, main_sub0_eps[0], main_eps1[ind0], main_tip1, main_tip2, main_eps2[ind0], main_sub0_eps[1]],
        "p6": [main_sub1_tip, main_sub1_eps[0], main_eps1[ind1], main_tip1, main_tip2, main_eps2[ind1], main_sub1_eps[1]],
    }), gr1
    

def sort_telement_roof_segments(self, roof_type, roof_id):

    roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
    segment_ids = list(roof["segment_id"])
    segment_planes = list(roof["plane_coefficients"])

    xs, ys, zs = list(roof["x_coordinates"]), list(roof["y_coordinates"]), list(roof["z_coordinates"])
    gr1 = GabledRoof(roof_type, roof_id, segment_ids, segment_planes, xs, ys, zs)
    return gr1
    
    intersections, gabled_matches = find_segment_matches(segment_ids, segment_planes, xs, ys, zs)

    main_roof_match = 0 if max(zs[gabled_matches[0][0]]) > max(zs[gabled_matches[1][0]]) else 1
    sub_roof_match = 1 if main_roof_match == 0 else 0

    minx_sub_seg0, maxx_sub_seg0, _, _, minz_sub_seg0, _ = find_min_max_values(xs[gabled_matches[sub_roof_match][0]], ys[gabled_matches[sub_roof_match][0]], zs[gabled_matches[sub_roof_match][0]])
    _, _, _, _, minz_sub_seg1, _ = find_min_max_values(xs[gabled_matches[sub_roof_match][1]], ys[gabled_matches[sub_roof_match][1]], zs[gabled_matches[sub_roof_match][1]])
    minx_main_seg0, maxx_main_seg0, _, _, minz_main_seg0, _ = find_min_max_values(xs[gabled_matches[main_roof_match][0]], ys[gabled_matches[main_roof_match][0]], zs[gabled_matches[main_roof_match][0]])
    _, _, _, _, minz_main_seg1, _ = find_min_max_values(xs[gabled_matches[main_roof_match][1]], ys[gabled_matches[main_roof_match][1]], zs[gabled_matches[main_roof_match][1]])
    
    global_zmin = min(minz_sub_seg0[2], minz_sub_seg1[2], minz_main_seg0[2], minz_main_seg1[2])

    closest_main_segment = 0 if distance(minz_sub_seg0, minz_main_seg0)+distance(minz_sub_seg1, minz_main_seg0) < distance(minz_sub_seg0, minz_main_seg1)+distance(minz_sub_seg1, minz_main_seg1) else 1

    main_sub_tip, main_sub_ips, main_sub_dvs = intersect_3planes(
        segment_planes[gabled_matches[sub_roof_match][0]], 
        segment_planes[gabled_matches[sub_roof_match][1]], 
        segment_planes[gabled_matches[main_roof_match][closest_main_segment]]
    )
    main_sub_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(main_sub_ips, main_sub_dvs) if abs(dv[2]) > 0.1]

    sub_edge_ref_point = minx_sub_seg0 if distance(minx_sub_seg0, main_sub_tip) > distance(maxx_sub_seg0, main_sub_tip) else maxx_sub_seg0
    sub_intersection_dv = intersections[sub_roof_match][1]
    sub_edge_plane = [sub_intersection_dv[0], sub_intersection_dv[1], 0, -sub_intersection_dv[0]*sub_edge_ref_point[0] - sub_intersection_dv[1]*sub_edge_ref_point[1] - 0*sub_edge_ref_point[2]]
    
    sub_tip, sub_ips, sub_dvs = intersect_3planes(
        segment_planes[gabled_matches[sub_roof_match][0]], 
        segment_planes[gabled_matches[sub_roof_match][1]], 
        sub_edge_plane
    )
    sub_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(sub_ips, sub_dvs) if abs(dv[2]) > 0.1]

    main_intersection_dv = intersections[main_roof_match][1]
    main_edge_plane1 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*minx_main_seg0[0] - main_intersection_dv[1]*minx_main_seg0[1] - 0*minx_main_seg0[2]]
    main_edge_plane2 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*maxx_main_seg0[0] - main_intersection_dv[1]*maxx_main_seg0[1] - 0*maxx_main_seg0[2]]

    main_tip1, main_ips1, main_dvs1 = intersect_3planes(
        segment_planes[gabled_matches[main_roof_match][0]], 
        segment_planes[gabled_matches[main_roof_match][1]], 
        main_edge_plane1
    )
    main_eps1 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips1, main_dvs1) if abs(dv[2]) > 0.1]

    main_tip2, main_ips2, main_dvs2 = intersect_3planes(
        segment_planes[gabled_matches[main_roof_match][0]], 
        segment_planes[gabled_matches[main_roof_match][1]], 
        main_edge_plane2
    )
    main_eps2 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips2, main_dvs2) if abs(dv[2]) > 0.1]

    ind = 1 if closest_main_segment==0 else 0
    return self.create_polygons(roof_type, {
        "p1": [main_sub_tip, main_sub_eps[0], sub_eps[0], sub_tip],
        "p2": [main_sub_tip, main_sub_eps[1], sub_eps[1], sub_tip],
        "p3": [main_sub_tip, main_sub_eps[0], main_eps1[ind], main_tip1, main_tip2, main_eps2[ind], main_sub_eps[1]],
        "p4": [main_tip1, main_tip2, main_eps1[closest_main_segment], main_eps2[closest_main_segment]]
    }), gr1
    
"""
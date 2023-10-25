from main import Lo2D
from plotter import Plotter

import geopandas as gpd
from shapely.ops import unary_union

def write_trondheim_buildings_to_csv():

    shape = gpd.read_file("./buildings/gis_osm_buildings_a_free_1.shp")
    trondheim_polygon = gpd.read_file("./Basisdata_5001_Trondheim_25832_Kommuner_GeoJSON.geojson").to_crs(4326).iloc[0].geometry
    trondheim_buildings = shape.within(trondheim_polygon)
    trondheim_buildings = shape.loc[trondheim_buildings].copy()
    trondheim_buildings.to_file("./buildings_trondheim.shp")

def merge_buildings(buildings, roofs_with_building, roofs_with_several_buildings):
    """
    Merge buildings when a roof has several buildings
    """
    roofs_several_buildings = roofs_with_building.query("roof_id in @roofs_with_several_buildings").sort_values(by=["roof_id", "classification", "index_right"])
    for roof in roofs_several_buildings.roof_id.unique():
        _roof = roofs_with_building.query("roof_id == @roof")
        building_indices = _roof.index_right.unique()
        _buildings = buildings.iloc[building_indices]
        merged = unary_union(_buildings.geometry.to_list())
        for b in _buildings.index:
            buildings.at[b, 'geometry'] = merged
    return buildings


if __name__ == "__main__":
    # write_trondheim_buildings_to_csv()

    trondheim_buildings = gpd.read_file("./buildings_trondheim")
    # trondheim_buildings.to_crs(4326)


    DATA_FOLDER_PATH = './data/roofs'

    lo2d = Lo2D(DATA_FOLDER_PATH)
    lo2d.fit()

    lo2d.roof_df.crs = 25832
    new_df = lo2d.roof_df.to_crs(trondheim_buildings.crs)

    matched_roof_segments_with_buildings = gpd.sjoin_nearest(new_df, trondheim_buildings)

    roofs_with_several_buildings = matched_roof_segments_with_buildings[["roof_id", "index_right"]].groupby("roof_id").describe().index_right.query("std > 0").index
    merged_buildings = merge_buildings(trondheim_buildings, matched_roof_segments_with_buildings, roofs_with_several_buildings)
    
    relevant_building_ids = matched_roof_segments_with_buildings.index_right.unique()
    relevant_buildings = merged_buildings.iloc[relevant_building_ids]

    matched_roofs_with_buildings = matched_roof_segments_with_buildings.merge(relevant_buildings, on="osm_id")
    matched_roofs_with_buildings = matched_roofs_with_buildings.drop(["geometry_x", "classification", "code_x", "fclass_x", "name_x", "type_x", "code_y", "fclass_y", "name_y", "type_y"], axis=1)
    matched_roofs_with_buildings = matched_roofs_with_buildings.drop_duplicates(subset='roof_id', keep='first')

    matched_roofs_with_buildings = gpd.GeoDataFrame(matched_roofs_with_buildings, geometry=list(matched_roofs_with_buildings["geometry_y"]))
    matched_roofs_with_buildings.crs = 4326
    matched_roofs_with_buildings = matched_roofs_with_buildings.to_crs(epsg=25832)

    for i in range(len(matched_roofs_with_buildings)):
        for obj in lo2d.roof_objects:
            if obj.roof_id == matched_roofs_with_buildings.iloc[i].roof_id:
                print(matched_roofs_with_buildings.iloc[i])
                # print(matched_roofs_with_buildings.iloc[i].crs)
                print(obj.roof_polygons)
                print()
                # for j in range(len(relevant_buildings)):
                #     if relevant_buildings.iloc[j].osm_id == matched_roofs_with_buildings.iloc[i].osm_id:
                Plotter.roof_polygons_with_building_footprint(
                    obj.roof_id,
                    obj.roof_polygons, 
                    matched_roofs_with_buildings.iloc[i].geometry,
                    obj.segment_xs, 
                    obj.segment_ys,
                    obj.segment_zs, 
                    10, 
                    'r'
                )
                break
        print()
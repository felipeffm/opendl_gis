import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import xarray as xr
import rasterio as r

# Folders paths
script_folder = os.path.dirname(__file__)
path_rasters = os.path.join(script_folder, "RGB - Brasil - Niteroi")
path_shp = os.path.join(script_folder, "lotes_pref.shp")
path_dataset = os.path.join(script_folder, "dataset")

# Raster projection
crs = rxr.crs.CRS.from_string('EPSG:31983')


def raster2dataset(path_shp, tarcol_shp='y_val', path_rasters=path_rasters, path_dataset=path_dataset, raster_crs=crs):

    # Filepaths
    metacsv_path = os.path.join(path_dataset, 'metadados_dataset.csv')
    path_rasters = [os.path.join(path_rasters, file_) for file_ in os.listdir(
        path_rasters) if file_.endswith('.tif')]

    # Cache
    metadados_imgs = []
    error = []

    # Load and subset shapefile
    crop_extent = gpd.read_file(path_shp)
    # ignore shapes without target
    crop_extent = crop_extent[~crop_extent[tarcol_shp].isnull()].copy()
    crop_extent.reset_index(inplace=True)

    # check dataset path
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)
    
    shape_name = path_shp.split('/')[-1].split('.')[0]

    # Load Rasters
    for path_raster in path_rasters:

        print(f"Raster {path_rasters.index(path_raster)+1} of {len(path_rasters)}")
        raster = rxr.open_rasterio(
            path_raster, 
            masked=True, 
            driver='GTiff', 
            mode='r', 
            chunks=True).squeeze()
        raster.rio.set_crs(crs, inplace=True)

        #Assure same projection as shapefile before crop raster
        assert crop_extent.crs == raster.rio.crs

        # Subset geometries(features) from shp inside the raster
        rc = raster.coords
        xmin, xmax = rc['x'].min().item(), rc['x'].max().item()
        ymin, ymax = rc['y'].min().item(), rc['y'].max().item()
        sub_extent = crop_extent.cx[xmin:xmax, ymin:ymax].copy()

        # Check if after the subset there are geometries inside the raster
        raster_contains_geom = len(sub_extent) > 0

        if raster_contains_geom:

            for i, r in tqdm(sub_extent.iterrows()):
                raster_name = path_raster.split('/')[-1].split('.')[0]

                geom = [r['geometry']]
                classe = r[tarcol_shp]
                filename = f"croppid_{r['index']}-class_{classe}-shp_{shape_name}-raster_{raster_name}.tif"
                
                # save each class in independet folder
                filepath = os.path.join(path_dataset, str(classe))
                img_abspath = os.path.join(filepath, filename)

                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                    assert os.path.exists(filepath) == True
                    print(f"folder to store images from class '{classe}' created")

                raster.rio.clip(geom, all_touched=True, drop=True,
                                from_disk=True).rio.to_raster(img_abspath)

                metadados_imgs.append({'filename': img_abspath, 'classe': classe, 'id': i,
                                       'data_processamento': datetime.now().strftime("%Hh %Mmin %m/%d/%y")})

        else:
            print('raster withour geom')
            error.append(path_raster)
            pass

        del raster

    # Load
    df_meta = pd.DataFrame(data=metadados_imgs)
    df_meta.to_csv(metacsv_path, index=False)
    print(f'Image storage finished, metadata stored in {metacsv_path}')
    print(f'Dataset saved in {path_dataset}')
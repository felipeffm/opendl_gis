import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
from datetime import datetime
from tqdm import tqdm

#Paths
path_raster = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/colorado-flood/spatial/boulder-leehill-rd/outputs/lidar_chm.tif"
path_shape = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/forma.shp"
path_base = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/1dataset"



#Load
raster = rxr.open_rasterio(path_raster, masked=True).squeeze()
crop_extent = gpd.read_file(path_shape)

for i in range(6):
    crop_extent = crop_extent.append(crop_extent)

def peneiradraster(raster,crop_extent,path_base,val_col_name = 'classif'):

    #local de armazenamento dos metadados do dataser
    metacsv_path = os.path.join(path_base,'metadados_dataset.csv')
    
    #Cache
    metadados_imgs = []

    #Raster and shape should be in same projection
    assert crop_extent.crs==raster.rio.crs

    counter=-1
    for i,r in tqdm(crop_extent.iterrows()):
        
        counter +=1

        #forma geoespacial
        geom = [r['geometry']]
        
        #classificação dos elementos dentro dessa forma
        classe = r[val_col_name]
        
        #nome do arquivo
        filename= f'cropped_id{i}_class{classe}_count{counter}.tif'

        #pasta para armazenar imagem recortada
        filepath = os.path.join(path_base,classe)
        
        #cria pasta caso não exista
        if not os.path.exists(filepath):
            os.mkdir(filepath)
            assert os.path.exists(filepath)==True
            print(f"pasta para armazenar imagens da classe '{classe}' criada")
        
        img_abspath = os.path.join(filepath,filename)
        
        
        raster_clipped = raster.rio.clip(geom)
        raster_clipped.rio.to_raster(img_abspath)

        metadados_imgs.append({'filename':img_abspath, 'classe': classe, 'id':i, 'data_processamento':datetime.now().strftime("%Hh %Mmin %m/%d/%y")})
    
    print(f'Armazenamento de imagens concluido, armazenado metadados em {metacsv_path}')
    #Load
    df_meta = pd.DataFrame(data = metadados_imgs)
    df_meta.to_csv(metacsv_path, index = False)

    print(f'Raster peneirado com sucesso, dataset e metadados salvos em {path_base}')

peneiradraster(raster,crop_extent,path_base,val_col_name = 'classif')

"""
#visualization
f, ax = plt.subplots(figsize=(10, 5))
lidar_chm_im.plot.imshow(ax=ax)
raster.plot(ax=ax, alpha=.8)
ax.set(title="Raster Layer with Shapefile Overlayed")
ax.set_axis_off()
plt.show()


f, ax = plt.subplots(figsize=(10, 4))
raster_clipped.plot(ax=ax)
ax.set(title="Raster Layer Cropped to Geodataframe Extent")
ax.set_axis_off()
plt.show()
"""
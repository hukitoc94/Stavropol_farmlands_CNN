import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import  show, adjust_band
from skimage.io import imread


import glob
from rasterio.features import shapes
from shapely.geometry import shape

import numpy as np
import pandas as pd
import json
from rasterio.plot import  show, adjust_band
import torch

import geopandas as gpd
from shapely import geometry



import os
import json
from tqdm import tqdm



def reproject_raster(raster_dir:str , raster_crs : str):
    """
    raster_dir - директория растра
    raster_crs - интересующая система в которую перекидываем 

    """
 
    with rasterio.open(f'./data/row_rasters/{raster_dir}') as src:
        src_transform = src.transform
        if raster_crs == src.crs:
            print("CRS is the same")
            return(raster_dir)

        # calculate the transform matrix for the output
        dst_transform, width, height = calculate_default_transform(
            src.crs,
            raster_crs,
            src.width,
            src.height,
            *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
        )
        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": raster_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": 0,  # replace 0 with np.nan
            }
        )

        with rasterio.open(f'./data/row_rasters/{raster_dir}', "w", **dst_kwargs) as dst:
            # iterate through bands
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=raster_crs,
                    resampling=Resampling.nearest,
                )

def clip_raster(raster_dir:str ,geometry,  fid,out_json_dir , out_pathces_dir, crs = None, m = False):
    """
    raster_dir - директория растра
    geometry - геометрия в которой все работает
    fid - номер квадрата
    m - если режете маску то true
    """
    if crs == None:
        crs = geometry.crs.to_epsg()
    with rasterio.open(raster_dir) as raster_data:
        out_img, out_transform = mask(dataset=raster_data,nodata = 0 ,shapes=geometry,crop=True)
        out_meta = raster_data.meta.copy()

        meta_dict = {"driver": 'GTiff',#"GTiff",JPEG
                        "height": 256,                   #out_img.shape[1],
                        "width":256,                    #out_img.shape[2], попробуем так чтобы все были одинаковые
                        "transform": out_transform,
                        "crs": crs,
                        "nodata" : 0 ,
                        'dtype' : rasterio.dtypes.uint16 #в формате флоат очень высокие значения и надо с этим что то делать 
                           # жто попробовать сразу привести все значения к 0-255 не знаю че выйдет 
                        }
        out_meta.update(meta_dict)

        entry = {"FID" : fid.item(), "metadata" : meta_dict, "file_location" : f"{out_pathces_dir}/{fid}.tif"} #тут надо подумать с директориями что делать
        with rasterio.open(f"{out_pathces_dir}/{fid}.tif", "w",**out_meta) as dest: # **out_meta
            dest.write(out_img)


        with open(out_json_dir, mode='r') as outfile:
            feeds = json.load(outfile)
        feeds['data'].append(entry)

        with open(out_json_dir ,mode='w') as outfile:
            json.dump(feeds, outfile)

""" 
вопрос что с этим куском делать)видимо надо как то переписать чтобы он универсально все делал 
if m: 
    arg = 'mask'
else:
    arg = 'image'
with open(f"data/Stavropol_dataSet/crs/{arg}_crs.json", mode='a') as fp:
    fp.write(json.dumps(entry, default=str))
    fp.write(',\n')


with rasterio.open(f"./data/Stavropol_dataSet/data/{arg}/{fid}.tif", "w",**out_meta) as dest: # **out_meta
    dest.write(out_img)
#else:
    #with open("data/Stavropol_dataSet/crs/image_crs.json", mode='a') as fp:
        #fp.write(json.dumps(entry))
        
    # with rasterio.open(f"./data/Stavropol_dataSet/data/image/{fid}.tif", "w", **out_meta) as dest:
        #dest.write(out_img)
    """
def get_raster_mask(raster, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    raster = adjust_band(raster)
    raster = raster.reshape(1,256,256,4)
    tensor_raster = torch.tensor(np.rollaxis(raster,3,1 ))
    tensor_raster = tensor_raster.to(torch.float32)
    tensor_raster = tensor_raster.to(device)
    out = model(tensor_raster)
    res = (out).detach().to('cpu')
    res = np.array(res)
    res = adjust_band(res)
    res.reshape(256,256)
    return(res > 0.62) #порог вообще надо причесать код везде иначе некрасиво 


def build_grid(vector,square_size = 2560, bias =  0):
    """
    vector - векторное представление той площади на которой мы будем строить решетку 
    square_size - размер квадрата 
    bias - смещение (перекрытие) квадратов
    
    output - коллекция из полигонов  square_size х square_size размером с перехлестом в размере bias
    """
    total_bounds = vector.total_bounds
    minX, minY, maxX, maxY = total_bounds
    x, y = (minX, minY)
    dstCrs = vector.crs.to_epsg()
    geom_array = []
    
    while y <= maxY:
        while x <= maxX:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size - bias
        x = minX
        y += square_size - bias
    
    fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(dstCrs)
    grid = fishnet.sjoin(vector)
    grid['fid'] = grid.index
    return(grid)



def clip_to_patches(raster_dir, geometry):

    pathes_dir = "./data/temp/raster_patches"
    json_dir = "./data/temp/patches_meta.json"
    crs =  geometry.crs.to_epsg()
    if not os.path.exists(pathes_dir):
        os.makedirs(pathes_dir)
        
    if not os.path.exists(json_dir):
        with open(json_dir, 'w') as outfile:  
            data ={"data" : []}
            json.dump(data, outfile)


    for i in tqdm(range(geometry.shape[0])):

        fid = geometry.iloc[i].fid
        geom = geometry.iloc[i].geometry
        clip_raster(raster_dir, [geom], fid, json_dir,pathes_dir, crs)


def build_mask_patches(meta_json_dir, model):   
    with open(meta_json_dir, mode='r') as outfile:
        feeds = json.load(outfile)
    if not os.path.exists("./data/temp/classified_patches/"):
        os.makedirs("./data/temp/classified_patches/")
    for i in tqdm(feeds["data"]):
        fid = i['FID']
        if not os.path.exists(f"./data/temp/classified_patches/{fid}.tif"): 

            patch_dir = i['file_location']

            raster = imread(patch_dir)
            out = get_raster_mask(raster, model)
            fid = i['FID']
            meta_data = i["metadata"]
            meta_data['count'] = 1
            meta_data["transform"] = meta_data["transform"][0:6]

        
            with rasterio.open(f"./data/temp/classified_patches/{fid}.tif", "w",**meta_data) as dest: # **out_meta
                dest.write(out.reshape(1,256,256))


def vectorize(binary_raster_dir):
    with rasterio.open(binary_raster_dir) as src:
        data = src.read(1, masked=True)

        # Use a generator instead of a list
        shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))

        # either build a pd.DataFrame
        df = pd.DataFrame(shape_gen, columns=['geometry', 'class'])
        gdf = gpd.GeoDataFrame(df["class"], geometry=df.geometry, crs=src.crs)
        return(gdf)



def get_vector_mask(json_dir, model, vector):

    print('применение сети для сегментации...')
    build_mask_patches(json_dir, model)

    
    search_criteria = "*.tif"
    q = os.path.join('./data/temp/classified_patches/', search_criteria)
    dem_fps = glob.glob(q)
    src_files_to_mosaic = []
    print('собираем общий растр...')
    for fp in tqdm(dem_fps):
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_raster_meta = {}

    out_raster_meta["height"] = mosaic.shape[1]
    out_raster_meta['width'] = mosaic.shape[2]
    out_raster_meta['transform'] = out_trans
    out_raster_meta["nodata"] = 0
    out_raster_meta["dtype"] = "uint8"
    out_raster_meta['crs'] =  vector.crs.to_epsg()
    out_raster_meta['driver'] = 'GTiff'
    out_raster_meta["transform"] = out_trans
    out_raster_meta['count'] = 1
    with rasterio.open("./data/temp/mosaic.tif", "w",**out_raster_meta) as dest: # **out_meta
        dest.write(mosaic)

    with rasterio.open("./data/temp/mosaic.tif") as dest: # **out_meta
        out_img, out_transform = mask(dataset=dest ,shapes=vector.geometry,nodata= 0, crop = True)

    out_raster_meta["height"] = out_img.shape[1]
    out_raster_meta['width'] = out_img.shape[2]
    out_raster_meta["transform"] = out_transform
    with rasterio.open("./data/temp/mosaic.tif", "w",**out_raster_meta) as dest: # **out_meta
            dest.write(out_img)
    result = vectorize("./data/temp/mosaic.tif")
    return(result)
# -*- coding: utf-8 -*-
'''
This code is used to download image from google.

original author: 
@date  : 2020-3-10
@author: Zheng Jie
@E-mail: zhengjie9510@qq.com

modification: 
@date  : 2021-10-18
@author: SeyyedHossein Hasanpour
'''
import os
import io
import math
from math import floor, pi, log, tan, atan, exp
import numpy as np
from threading import Thread, Lock
import urllib.request as ur
import PIL.Image as pil

import cv2
from osgeo import gdal, osr
import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Google Map Downloader 1.1.2')

parser.add_argument('-s',
                    '--server',
                    metavar='server',
                    type=str,
                    default='Google',
                    help='The download server (Google, Google China)')

parser.add_argument('-z',
                    '--zoom',
                    metavar='zoom-level',
                    type=int,
                    default=15,
                    choices=[n for n in range(1, 22)],
                    help='The zoom level at which you want the satallite images (1-21)')

parser.add_argument('-st',
                    '--style',
                    metavar='style',
                    type=str,
                    default='s',
                    choices=['s', 'm', 'y', 't', 'p', 'h'],
                    help='The map style (s:satellite, m:map, y:satellite with label, t:terrain, p:terrain with label, h:label)')

parser.add_argument('-p',
                    '--path',
                    metavar='path',
                    type=str,
                    default='map.tif',
                    help='The file path you want to save as. if you want to use save-memory option, choose a fast drive (e.g. an ssd) otherwise,\
                        based on how large the area and zoom level of your choice is, it will take a lot of time.')

parser.add_argument('-tl',
                    '--area_coordinate_top_left',
                    metavar='top_left',
                    type=float,
                    nargs="+",
                    required=True,
                    help="The area's top left corner latitue and longitude values (e.g. 35.798443 51.155698 )")

parser.add_argument('-br',
                    '--area_coordinate_bottom_right',
                    metavar='bottom_right',
                    type=float,
                    nargs="+",
                    required=True,
                    help="The area's bottom right corner latitue and longitude values (e.g. 35.542383 51.602294 )")

parser.add_argument('-sm',
                    '--save_memory',
                    default=False,
                    action="store_true",
                    help="Whether to conserve/save computer memory(RAM) as much as possible.\
                    This is important for saving large areas or when you have little RAM available.")

parser.add_argument('-km',
                    '--keep_rgb_map',
                    default=True,
                    action="store_true",
                    help="Whether to keep(save) the intermediate map in jpeg as well")

parser.add_argument('-kt',
                    '--keep_tiles',
                    default=True,
                    action="store_true",
                    help="Whether to keep(save) the tiles as well")

parser.add_argument('-tp',
                    '--tiles_save_path',
                    default='tiles/',
                    help="The directory path inside which to save the tiles")

parser.add_argument('-ut'
                    ,'--use_existing_tiles',
                    action="store_true",
                    help="Whether to use an existing tiles folders(uses tiles_save_path)")

parser.add_argument('-kc'
                    ,'--keep_cache',
                    action="store_true",
                    help="Whether to keep a numpy cache (this is only valid if you have used save_memory)")                    

args = parser.parse_args()

# ------------------Interchange between WGS-84 and Web Mercator-------------------------
# WGS-84 to Web Mercator
def wgs_to_mercator(y, x):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return y2, x2

# Web Mercator to WGS-84
def mercator_to_wgs(y, x):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return y2, x2


# -------------------------------------------------------------

# -----------------Interchange between GCJ-02 to WGS-84---------------------------
# All public geographic data in mainland China need to be encrypted with GCJ-02, introducing random bias
# This part of the code is used to remove the bias
def transformLat(y, x):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(y, x):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    ''' 
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: Projection factor of satellite ellipsoidal coordinates projected onto a flat map coordinate system
    ee = 0.00669342162296594323  # ee: Eccentricity of ellipsoid
    dLat = transformLat(lat - 35.0, lon - 105.0)
    dLon = transformLon(lat - 35.0, lon - 105.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}


def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False


def gcj_to_wgs(gcjLat, gcjLon):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLat, gcjLon)
    d = delta(gcjLat, gcjLon)
    return (gcjLat - d["lat"], gcjLon - d["lon"])


def wgs_to_gcj(wgsLat, wgsLon):
    if outOfChina(wgsLat, wgsLon):
        return wgsLat, wgsLon
    d = delta(wgsLat, wgsLon)
    return  wgsLat + d["lat"], wgsLon + d["lon"]


# --------------------------------------------------------------

# ---------------------------------------------------------
# Get tile coordinates in Google Maps based on latitude and longitude of WGS-84
def wgs_to_tile( latitude, longitude, zoom):
    """Get google-style tile cooridinate from geographical coordinate

    Note: 
    Note that this will return a tile coordinate. and a tile coordinate
    is located at (0,0) or origin of a tile. 
    all tiles are 256x256. there are as many gps locations inside of a tile 
    that when used with this function will return the same exact tile coordinate
    (as they all reside in that tile obviously) in order to get the displacement 
    of your gps location, after calculating the tile coordinate, calculate tile's 
    (0,0) gps location from the newly acgieved tile coordinate. then you have gps location
    at (0,0) and now can calculate howfar your initial gps location is from the 
    origin of the tile. 

    Args:
        latitude (float): Latitude
        longitude (float): Longittude
        zoom (int): zoom

    Raises:
        TypeError: [description]
        TypeError: [description]

    Returns:
        tuple(int,int): returns tile coordinate in the form of (x,y)
    """
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(longitude) and isnum(latitude)):
        raise TypeError("latitude and longitude must be int or float!")

    if not isinstance(zoom, int) or zoom < 0 or zoom > 22:
        raise TypeError("zoom must be int and between 0 to 22.")

    if longitude < 0:
        longitude = 180 + longitude
    else:
        longitude += 180
    longitude /= 360  # make longitude to (0,1)

    latitude = 85.0511287798 if latitude > 85.0511287798 else latitude
    latitude = -85.0511287798 if latitude < -85.0511287798 else latitude
    latitude = math.log(math.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)
    latitude /= 180  # make latitude to (-1,1)
    latitude = 1 - (latitude + 1) / 2  # make latitude to (0,1) and left top is 0-point

    num = 2 ** zoom
    y = math.floor(latitude * num)
    x = math.floor(longitude * num)
    return y, x


def pixls_to_mercator(zb):
    # Get the web Mercator projection coordinates of the four corners of the area according to the four corner coordinates of the tile
    iny, inx = zb["TL"]  # top left 
    iny2, inx2 = zb["BR"]  # bottom right 
    length = 20037508.3427892
    sum = 2 ** zb["z"]
    TLx = inx / sum * length * 2 - length
    TLy = -(iny / sum * length * 2) + length

    BRx = (inx2 + 1) / sum * length * 2 - length
    BRy = -((iny2 + 1) / sum * length * 2) + length

    # TL=Top Left, BR=Buttom Right 
    # Returns the projected coordinates of the four corners
    res = {'TL': (TLy, TLx), 'BR': (BRy, BRx),
           'BL': (BRy, TLx), 'TR': (TLy, BRx)}
    return res


def tile_to_pixls(zb):
    # Tile coordinates are converted to pixel coordinates of the four corners
    out = {}
    width = (zb["TR"][1] - zb["TL"][1] + 1) * 256
    height = (zb["BL"][0] - zb["TL"][0] + 1) * 256
    out["TL"] = (0, 0)
    out["TR"] = (0, width)
    out["BL"] = (-height, 0)
    out["BR"] = (-height, width)
    return out


# -----------------------------------------------------------
#TODO: 
# add support for downloading tiles
# add support for conserving memory  
# ---------------------------------------------------------
class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, num_threads, total_thread_count, urls, datas, update, conserve_memory, save_tiles, tile_save_path):
        # num_threads represents the number of threads
        # total_thread_count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.num_threads = num_threads
        self.total_thread_count = total_thread_count
        self.update = update
        self.save_tiles = save_tiles
        self.tile_save_path = tile_save_path
        self.conserve_memory = conserve_memory

    def download(self, tile_url, tile_coordx, tile_coordy):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
        header = ur.Request(tile_url, headers=HEADERS)
        err = 0
        while (err < 3):
            try:
                data = ur.urlopen(header).read()
                try:
                    if self.save_tiles or self.conserve_memory:
                        img_blob = io.BytesIO(data)
                        tile_img = pil.open(img_blob)
                        tile_img.save(f'{self.tile_save_path}/tile_{tile_coordy}_{tile_coordx}.jpg')
                except Exception as ex:
                    print(f'An exception has occured during saving the tiles:\n{ex}')
                    exit(-1)
            except:
                err += 1
            else:
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.total_thread_count != self.num_threads:
                continue
            
            if self.conserve_memory:
                self.download(*url)
            else:
                self.datas[i] = self.download(*url)

            if mutex.acquire():
                self.update()
                mutex.release()


# ---------------------------------------------------------

# ---------------------------------------------------------
#TODO: Add more servers

MAP_URLS = {
    "Google": "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}",
    "Google China": "http://mt2.google.cn/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}"}

def get_url(source, x, y, z, style):  #
    if source == 'Google China':
        url = MAP_URLS["Google China"].format(x=x, y=y, z=z, style=style)
    elif source == 'Google':
        url = MAP_URLS["Google"].format(x=x, y=y, z=z, style=style)
    else:
        raise Exception("Unknown Map Source ! ")
    return url,x,y


def get_urls(y1, x1, y2, x2, z, source='google', style='s'):
    pos1y, pos1x  = wgs_to_tile(y1, x1, z)
    pos2y, pos2x  = wgs_to_tile(y2, x2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("-Total tiles number(YxX): {y} x {x}".format(y=leny, x=lenx))
    urls = [get_url(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]
    return urls

# ---------------------------------------------------------

# ---------------------------------------------------------
def download_tiles(urls, multi=10, conserve_memory=False, save_tiles=False, tiles_save_path='tiles', use_existing_tiles=False):
    def makeupdate(s):
        def up():
            global COUNT
            COUNT += 1
            print("\r--DownLoading...[{0}/{1}]".format(COUNT, s), end='')

        return up
    
    url_len = len(urls)
    datas = [None] * url_len

    if use_existing_tiles:
        if os.path.exists(tiles_save_path) and len(os.listdir(tiles_save_path))>0:

            if not conserve_memory:
                # load datas!
                print(f'--Loading tiles into memory:...')
                for i, img_fname in enumerate(tqdm(sorted(os.listdir(tiles_save_path)))):
                    img_path = os.path.join(tiles_save_path,img_fname)
                    img = pil.open(img_path)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    datas[i] = img_byte_arr.getvalue()
            return datas 
        else:
            print(f"Error: The tile directory '{tiles_save_path}' doesnt exist or does not have any tiles.")
            exit(-1)

    if save_tiles or conserve_memory:
        if (os.path.exists(tiles_save_path) and len(os.listdir(tiles_save_path)) >0):
            print(f"Error: The path '{tiles_save_path}' is full! please choose an empty directory.")
            exit(-1)
        else:
            os.makedirs(tiles_save_path, exist_ok=False)
        
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas, makeupdate(url_len), 
                        conserve_memory=conserve_memory, 
                        save_tiles=save_tiles, 
                        tile_save_path=tiles_save_path) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()
    return datas


def merge_tiles(datas, y1, x1, y2, x2, z, file_path, save_memory, save_map, cache_dir):
    print('\n--Merging downloaded tiles...')
    pos1y, pos1x= wgs_to_tile(y1, x1, z)
    pos2y, pos2x= wgs_to_tile(y2, x2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1

    # we should use PNG as JPEG can not handle more thabn 64K pixels!
    # becasue image formats like .jpeg only have 2 bytes available to 
    # record the width and height, and 2 bytes has a max decimal value of 65,535.
    # unfortunately this incurs a lot of overhead we have to deal with
    # anyway! and resort to png which has 4 bytes for this (instead of jpegs 2 bytes)
    # this allows for 2^32 or 4,294,967,296 pixels
    # ref https://communities.efi.com/s/article/Error-JPEG-Compression-failed?language=en_US
    # This is caused by a combination of preview DPI setting and the height or width of the output file. 
    # The maximum pixel height/width of a JPEG is 65535 or 2^16-1. Many software implementations reduce 
    # this slightly to 65500 including libjpeg and libjpeg-turbo. This is a limitation of the JPEG standard.
    map_img_path = file_path.replace(os.path.splitext(file_path)[1],'.png')
    map_img_cache = file_path.replace(os.path.splitext(file_path)[1],'_memmap.np')
    if save_memory:
        dim_x = 256
        dim_y = 256
        nrows = leny*256
        ncols = lenx*256
        nchannels = 4
        # we sort them as the order of tiles matter
        tiles_list = sorted(os.listdir(cache_dir))
        map_img_raw = np.memmap(map_img_cache, dtype=np.uint8, mode='w+', shape=(nrows, ncols, nchannels))
        print(f'Note: Depending on your image size and your HDD/SSD transfer speed, this may take a long time!')        
        for i, img_name in enumerate(tqdm(tiles_list)):
            img_path = os.path.join(cache_dir, img_name)
            y, x = i // lenx, i % lenx
            start_x = x * dim_x
            start_y = y * dim_y
            end_x = start_x + dim_x
            end_y = start_y + dim_y
            img = pil.open(img_path).convert('RGBA')
            map_img_raw[start_y:end_y, start_x:end_x, :] = np.asarray(img)
            # print(f'pos: ({start_x},{start_y}) - ({end_x},{end_y})')
        del map_img_raw
        print('\n--Tiles merge completed')
        map_img_raw = np.memmap(map_img_cache, dtype=np.uint8, mode='r', shape=(nrows, ncols, nchannels))
        if save_map:
            print(f'--Map image is being saved...')
            pil.fromarray(map_img_raw, mode='RGBA').save(map_img_path)
            print(f'--Map image is saved!')
        # del map_img_raw
        # os.remove('map_img_memmap.np')
        # print(f'map_img_raw shape: {map_img_raw.shape}')
        return map_img_raw
    
    else:
        map_img = pil.new('RGBA', (lenx * 256, leny * 256))
        for i, data in enumerate(tqdm(datas)):
            tile_bytes = io.BytesIO(data)
            tile_img = pil.open(tile_bytes)

            y, x = i // lenx, i % lenx
            map_img.paste(tile_img, (x * 256, y * 256))
    
        print('\n--Tiles merge completed')
        # Save the map image
        if save_map:
            print(f'--Map image is being saved...')
            map_img.save(map_img_path)
            print(f'--Map image is saved!')
        return np.array(map_img)


# ---------------------------------------------------------

# ---------------------------------------------------------
def getExtent(y1, x1, y2, x2, z, source="Google China"):
    pos1y, pos1x = wgs_to_tile(y1, x1, z)
    pos2y, pos2x = wgs_to_tile(y2, x2, z)
    Xframe = pixls_to_mercator({"TL": (pos1y, pos1x), "TR": (pos2y, pos1x), "BL": (pos1y, pos2x), "BR": (pos2y, pos2x), "z": z})
    for i in ["TL", "BL", "TR", "BR"]:
        Xframe[i] = mercator_to_wgs(*Xframe[i])
    if source == "Google":
        pass
    elif source == "Google China":
        for i in ["TL", "BL", "TR", "BR"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: source.")
    return Xframe


def saveTiff(r, g, b, gt, filePath):
    print(f'\n-Saving the final tif image...')
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    # print(f' r.shape: { r.shape} filePath: {filePath}')
    dset_output = driver.Create(filePath, r.shape[1], r.shape[0], 3, gdal.GDT_Byte)
    dset_output.SetGeoTransform(gt)
    try:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)
        dset_output.SetSpatialRef(proj)
    except:
        print("Error: Coordinate system setting failed")
    dset_output.GetRasterBand(1).WriteArray(r)
    dset_output.GetRasterBand(2).WriteArray(g)
    dset_output.GetRasterBand(3).WriteArray(b)
    dset_output.FlushCache()
    dset_output = None
    print("--Image Saved")


# ---------------------------------------------------------

def main(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, file_path, style='s', server="Google", 
         save_memory=False, keep_cache=False, keep_rgb_map=False, save_tiles=False, tile_save_path='tiles', use_existing_tiles=False):
    """Downloads a map of the area specified by the given coordinates.

    Args:
        top_left_lat (float): The top left corner latitude of the area you would like to capture
        top_left_lon (float): The top left corner longitude of the area you would like to capture
        bottom_right_lat (float): The bottom right corner latitude of the area you would like to capture
        bottom_right_lon (float): The bottom right corner longitude of the area you would like to capture
        zoom (int): The zoom level at which you want the satallite images (1-21)
        file_path (str): The file path you want to save as
        style (str, optional): The map style (s:satellite, m:map, y:satellite with label, t:terrain, p:terrain with label, h:label). Defaults to 's'.
        server (str, optional): The download server (Google, Google China). Defaults to "Google".
        save_memory (bool, optional): Whether to conserve/save computer memory(RAM) as much as possible.\
                    This is important for saving large areas or when you have little RAM available. Defaults to False.
        keep_cache (bool, optional): Whether to keep numpy cache which is temporarily created when save_memory is used. Defaults to False.
        keep_rgb_map (bool, optional): Whether to keep(save) the intermediate map in jpeg as well. Defaults to False.
        save_tiles (bool, optional): Whether to keep(save) the tiles as well. Defaults to False.
        tile_save_path (str, optional): The directory path inside which to save the tiles. Defaults to 'tiles'.
        use_existing_tiles (bool, optional): Whether to use an existing tiles folders(uses tiles_save_path). Defaults to False.
    """
    # ---------------------------------------------------------
    # Get the urls of all tiles in the extent
    urls = get_urls(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, server, style)

    # Download tiles
    datas = download_tiles(urls, multi=10,  
                           conserve_memory=save_memory, 
                           save_tiles=save_tiles, 
                           tiles_save_path=tile_save_path,
                           use_existing_tiles=use_existing_tiles)

    # Combine downloaded tile maps into one map
    if save_memory and not save_tiles:
        cache_dir = 'cache'
    else:
        cache_dir = tile_save_path

    map_img = merge_tiles(datas, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, file_path, 
                         save_memory=save_memory,
                         save_map=keep_rgb_map,
                         cache_dir=cache_dir)

    rows,cols = map_img[:,:,0].shape
    print(f'Map shape: {map_img.shape}')

    # Get the spatial information of the four corners of the merged map and use it for outputting
    extent = getExtent(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, server)
    # gt = (extent['TL'][0], (extent['BR'][0] - extent['TL'][0]) / r.shape[1], 0, extent['TL'][1], 0,
    #       (extent['BR'][1] - extent['TL'][1]) / r.shape[0])
    gt = (extent['TL'][1], (extent['BR'][1] - extent['TL'][1]) / rows, 0, extent['TL'][0], 0,
          (extent['BR'][0] - extent['TL'][0]) / cols)

    saveTiff(map_img[:,:,0], map_img[:,:,1], map_img[:,:,2], gt, file_path)
    
    # todo: remove this in a better way!
    if save_memory and not keep_cache:
        map_img_cache_path = file_path.replace(os.path.splitext(file_path)[1], '_memmap.np')
        if os.path.exists(map_img_cache_path):
            del map_img
            os.remove(map_img_cache_path)

# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    COUNT = 0  # Progress display, starting at 0
    mutex = Lock()

    style_description = {'s':'sattelite', 'm':'map', 'y': 'satellite with label', 't':'terrain', 'p': 'terrain with labels', 'h': 'label'}
    
    print(f'-Downloading the map using the following information:')
    print(f'--zoom:                     {args.zoom}')
    print(f'--style:                    {args.style} ({style_description[args.style]})')
    print(f'--server:                   {args.server}')
    print(f'--file path:                {args.path}')
    print(f'--save memory               {args.save_memory}')
    print(f'--Keep np cache:            {args.keep_cache}')
    print(f'--save tiles:               {args.keep_tiles}')
    print(f'--save rgb map:             {args.keep_rgb_map}')
    print(f'--use existing tiles:       {args.use_existing_tiles}')
    print(f'--tiles save location:      {args.tiles_save_path}')
    print(f'--area top left corner:     {args.area_coordinate_top_left}')
    print(f'--area bottom right corner: {args.area_coordinate_bottom_right}\n')
        
    top_left_lat, top_left_lon = tuple(args.area_coordinate_top_left)
    bottom_right_lat, bottom_right_lon = tuple(args.area_coordinate_bottom_right)

    main(top_left_lat=top_left_lat, 
         top_left_lon=top_left_lon,
         bottom_right_lat=bottom_right_lat, 
         bottom_right_lon=bottom_right_lon,
         zoom=args.zoom, 
         file_path=args.path, 
         style=args.style,
         server=args.server,
         save_memory=args.save_memory,
         keep_cache=args.keep_cache,
         keep_rgb_map=args.keep_rgb_map,
         save_tiles=args.keep_tiles, 
         tile_save_path=args.tiles_save_path,
         use_existing_tiles=args.use_existing_tiles)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f'\n-Elapsed time: {elapsed:.2f} seconds')

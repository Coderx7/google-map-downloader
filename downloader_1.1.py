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

import io
import math
import numpy as np
from math import floor, pi, log, tan, atan, exp
from threading import Thread, Lock
import urllib.request as ur
import PIL.Image as pil

import cv2
from osgeo import gdal, osr
import time
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
                    help='The map style (s:satellite, m:map, y:satellite with label, t:terrain, p:terrain with label, h:label')

parser.add_argument('-p',
                    '--path',
                    metavar='path',
                    type=str,
                    default='map.tif',
                    help='The file path you want to save as')

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

# ---------------------------------------------------------
class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas, update):
        # index represents the number of threads
        # count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count
        self.update = update

    def download(self, url):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
        header = ur.Request(url, headers=HEADERS)
        err = 0
        while (err < 3):
            try:
                data = ur.urlopen(header).read()
            except:
                err += 1
            else:
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)
            if mutex.acquire():
                self.update()
                mutex.release()


# ---------------------------------------------------------

# ---------------------------------------------------------
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
    return url


def get_urls(y1, x1, y2, x2, z, source='google', style='s'):
    pos1y, pos1x  = wgs_to_tile(y1, x1, z)
    pos2y, pos2x  = wgs_to_tile(y2, x2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Total tiles number(YxX)：{y} X {x}".format(y=leny, x=lenx))
    urls = [get_url(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]
    return urls


# ---------------------------------------------------------

# ---------------------------------------------------------
def download_tiles(urls, multi=10):
    def makeupdate(s):
        def up():
            global COUNT
            COUNT += 1
            print("\rDownLoading...[{0}/{1}]".format(COUNT, s), end='')

        return up

    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas, makeupdate(url_len)) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()
    return datas


def merge_tiles(datas, y1, x1, y2, x2, z):
    pos1y, pos1x= wgs_to_tile(y1, x1, z)
    pos2y, pos2x= wgs_to_tile(y2, x2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)

        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))
    print('\nTiles merge completed')
    return outpic


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
    fname_out = filePath
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    dset_output = driver.Create(fname_out, r.shape[1], r.shape[0], 3, gdal.GDT_Byte)
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
    print("Image Saved")


# ---------------------------------------------------------

def main(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, filePath, style='s', server="Google China"):
    """
    Download images based on spatial extent.

    East longitude is positive and west longitude is negative.
    North latitude is positive, south latitude is negative.

    Parameters
    ----------
    top_left_lat, top_left_lon : top(lat)-left(lon) coordinate, for example (38.866, 100.361)
        
    bottom_right_lat, bottom_right_lon : bottom(lat)-right(lon) coordinate
        
    z : zoom

    filePath : File path for storing results, TIFF format
        
    style : 
        m for map; 
        s for satellite; 
        y for satellite with label; 
        t for terrain; 
        p for terrain with label; 
        h for label;
    
    source : Google China (default) or Google
    """
    # ---------------------------------------------------------
    # Get the urls of all tiles in the extent
    urls = get_urls(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, server, style)

    # Download tiles
    datas = download_tiles(urls)

    # Combine downloaded tile maps into one map
    outpic = merge_tiles(datas, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom)
    outpic = outpic.convert('RGB')
    r, g, b = cv2.split(np.array(outpic))

    # Get the spatial information of the four corners of the merged map and use it for outputting
    extent = getExtent(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, zoom, server)
    gt = (extent['TL'][1], (extent['BR'][1] - extent['TL'][1]) / r.shape[0], 0, extent['TL'][0], 0,
          (extent['BR'][0] - extent['TL'][0]) / r.shape[1])
    # print(f'gt: {gt}')
    saveTiff(r, g, b, gt, filePath)


# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    COUNT = 0  # Progress display, starting at 0
    mutex = Lock()

    style_description = {'s':'sattelite', 'm':'map', 'y': 'satellite with label', 't':'terrain', 'p': 'terrain with labels', 'h': 'label'}

    print(f'Downloading the map using the following information:\n')
    print(f'zoom:                     {args.zoom}')
    print(f'style:                    {args.style} ({style_description[args.style]})')
    print(f'server:                   {args.server}')
    print(f'file path:                {args.path}')
    print(f'area top left corner:     {args.area_coordinate_top_left}')
    print(f'area bottom right corner: {args.area_coordinate_bottom_right}\n')
        
    top_left_lat, top_left_lon = tuple(args.area_coordinate_top_left)
    bottom_right_lat, bottom_right_lon = tuple(args.area_coordinate_bottom_right)

    main(top_left_lat=top_left_lat, 
         top_left_lon=top_left_lon,
         bottom_right_lat=bottom_right_lat, 
         bottom_right_lon=bottom_right_lon,
         zoom=args.zoom, 
         filePath=args.path, 
         style=args.style,
         server=args.server)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f'lasted a total of {elapsed:.2f} seconds')

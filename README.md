# Google-Map-Downloader
![Google-Map-Downloader](https://geospatialmedia.s3.amazonaws.com/wp-content/uploads/2016/07/google-earth.jpg)
A small tool for downloading google map images. just specify the area you need by two sets of coordinates and you are good to go!
I tried to add several options to make it even easier to use including an option to save RAM at the expense of speed!
This is usually required for large areas (images with 60000x84000 pixels!(my usecase) and more). 

## Usage: 
Basically you can do something like this: 
```python
 python downloader.py -tl  35.798443 51.155698 -br 35.542383 51.602294 -p map-17.tiff -z 17 -kt -sm -tp tiles_17 
```
to know what these switches mean, simply see the help by running:
 ```python
 python downloader.py -h
 ```
for a list of options.

## Issues
If you encounter the problem of Bad network link, you can change the HEADERS in the download function, and try again.
```python
def download(self,url):
        HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.76 Safari/537.36'}
        header = ur.Request(url,headers=HEADERS)
        err=0
        while(err<3):
            try:
                data = ur.urlopen(header).read()
            except:
                err+=1
            else:
                return data
        raise Exception("Bad network link.")
```
##Note:
This is based on the good work of [zhengjie9510](https://github.com/zhengjie9510/google-map-downloader) 
"""As simple as it gets - but probably not too accurate"""

from tesserocr import PyTessBaseAPI

img = "path"

with PyTessBaseAPI(psm=6) as api:
    api.SetImageFile(img)
    api.GetThresholdedImage().show()
    print api.GetUTF8Text()

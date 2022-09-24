#Fotoğraf dosya uzantılarını değiştirmek için kullandığımız script
from PIL import Image
import glob, os

current_dir = os.path.dirname("C:/YOLO/images/suni_hasar/")

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):

    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(title + ext)

    im1 = Image.open(pathAndFilename)
    im1.save("C:/YOLO/images/suni_hasar/yeni/" +  title + ".png")
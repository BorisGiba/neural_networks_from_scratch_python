from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def save():
    global image_number
    filename = f'ownImage_{image_number}.png'   # image_number increments by 1 at every save
    image1.resize((28,28), Image.ANTIALIAS).save(filename)
    pix_val=getData()
    npix_val=normaliseData(pix_val)
    npix_val=np.array(npix_val)
    np.save("OwnImageData", npix_val)

    
def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=28)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=28)
    lastx, lasty = x, y

def drawImage():
    global cv,root,lastx,lasty,draw,image_number,image1
    root = Tk()

    lastx, lasty = None, None
    image_number = 0

    cv = Canvas(root, width=280, height=280, bg='white')
    # --- PIL
    image1 = PIL.Image.new('RGB', (280, 280), 'white')
    draw = ImageDraw.Draw(image1)

    cv.bind('<1>', activate_paint)
    cv.pack(expand=YES, fill=BOTH)

    #btn_save = Button(text="save", command=lambda: [save(),cb()])
    btn_save = Button(text="save", command=save)
    btn_save.pack()

    root.mainloop()



def getData(path="ownImage_0.png"):
    im = Image.open(path)
    #im_grey = im.convert('LA') # convert to grayscale
    #width, height = im.size
    pix_val=list(im.getdata())

    return pix_val

def normaliseData(data):
    xmin=255
    xmax=0
    singleValues=[]
    normalisedData=[]
    for valueTuple in data:
        singleValue=np.amax(valueTuple)
        singleValues.append(singleValue)

    for value in singleValues:
        normalisedValue=(value-xmin) / (xmax-xmin)
        normalisedData.append(normalisedValue)

    return normalisedData

def showImage():
    a=getData()
    b=normaliseData(a)
    b=np.array(b)
    label=0
    image=b.reshape([28,28])
    plt.imshow(image, cmap=plt.get_cmap("gray_r"))
    plt.title('Example: %d  Label: %d' % (0, label))
    plt.show()
    
    
    

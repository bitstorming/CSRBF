if __name__ == "__main__":
    from skimage.io import imread, imsave
    from Transform import TPSTransform, CSRBFTransform

    img = imread("boats_grid.jpg")
    h, w = img.shape
    anchor = [[0,0], [w, 0], [0, h], [w, h]] # anchor at four corner of image

    # expand and shrink at two location
    src = [[145, 130], [220, 130], [145, 180], [220, 180], [410, 317], [604, 314], [413, 471], [605, 472]]
    dst = [[115, 90],  [265, 90],  [115, 245], [270, 245], [445, 350], [570, 350], [445, 445], [570, 445]]
    src += anchor
    dst += anchor

    tps = TPSTransform()
    warped = tps.transform(img, src, dst, order=0)
    imsave("boats_tps.jpg", warped)

    csrbf = CSRBFTransform(200)
    warped = csrbf.transform(img, src, dst, order=0)
    imsave("boats_csrbf.jpg", warped)
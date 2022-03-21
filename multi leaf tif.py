import cv2

ret, images = cv2.imreadmulti('rgbc.tif')
# cv2.imshow('', images[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(images[1])

im = cv2.imread('practice image.tif')
cv2.imshow('', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(im)
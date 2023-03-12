import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

traj = cv.VideoCapture('ball.mov')
x = []
y = []
pts = []
z = 0
last_x = 0


def least_sqaure(x, y):
    Y = np.matrix([[i for i in y]])
    Y = np.transpose(Y)
    X_t = np.matrix([[i**2 for i in x], [i for i in x], [1 for i in x]])
    X = np.transpose(X_t)
    B = np.linalg.inv((X_t @ X)) @ ((X_t @ Y))
    a = float(B[0])
    b = float(B[1])
    c = float(B[2])
    return a, b, c


while (True):
    ret, frame = traj.read()
    if not ret:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 175, 120])
    upper_red = np.array([255, 255, 240])

    mask = cv.inRange(hsv, lower_red, upper_red)
    res = cv.bitwise_and(frame, frame, mask=mask)

    coordinates = (np.column_stack(np.where(mask != 0)))
    
    if coordinates[:, 1].size != 0 and coordinates[:, 0].size != 0:
        x_1, y_1 = (np.sum(coordinates[:, 1])/coordinates[:, 1].size,
                    np.sum(coordinates[:, 0])/coordinates[:, 0].size)
        x.append(x_1)
        y.append(y_1)
        center = (x_1, y_1)
        pts.append(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv.line(frame, np.asarray(
                pts[i - 1], dtype=int), np.asarray(pts[i], dtype=int), (255, 255, 255), 1)
            # cv.circle(frame, np.asarray(pts[i], dtype = int), 10, (0, 255, 255), 1)
    cv.imshow('frame', frame)
    # cv.imshow('mask',mask)
    # cv.imshow('res',res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
traj.release()
plt.scatter(x, y, s=3, label='Pixel coordinates of the center point of the ball')

first_y = y[0]

a, b, c = least_sqaure(x, y)
print('Equation of the curve, y = ', a, 'x\u00b2 ', b, 'x +', c)
m = []
g, f = np.roots([a, b, (c-first_y-300)])
if g > 0:
    print('x-coordinate of the ball’s landing spot in pixels: ', int(g))
elif f > 0:
    print('x-coordinate of the ball’s landing spot in pixels: ', int(f))
for i in x:
    m.append(a*(i**2) + b*i + c)
plt.plot(x, m, 'r', label='Best fit curve using least squares')
plt.legend()
plt.show()

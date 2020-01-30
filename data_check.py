import numpy as np
import matplotlib.pyplot as plt
import cv2

class IndexTracker(object):
    def __init__(self, ax, X, title='CTs and Contours'):
        self.ax = ax
        ax.set_title(title)
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class display_contours(object):
    def __init__(self, cts, contours, fill=False, threshold=10):
        self.contours = 255.*contours
        self.cts = cts*255.

        if fill:
            for j in range(self.contours.shape[0]):
                th, ob_th = cv2.threshold(self.contours[j,:,:].astype('uint8'), threshold, 255, cv2.THRESH_BINARY)
                ob_floodfill = ob_th.copy()
                h, w = ob_th.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(ob_floodfill, mask, (0,0), 255)
                ob_floodfill_inv = cv2.bitwise_not(ob_floodfill)
                ob_out = ob_th | ob_floodfill_inv
                self.contours[j,:,:] = ob_out

    def plot_masks(self):
        stacked = np.concatenate((self.cts, self.contours), axis=2)
        fig, axs = plt.subplots(1, 1)
        tracker = IndexTracker(axs, np.moveaxis(stacked,0,2))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

        return True

    def plot_contours(self, thresh_val=127, levels=255, window=True):
        ct_copy = self.cts.copy()
        for i in range(ct_copy.shape[0]):
            _, thresh = cv2.threshold(self.contours[i,:,:].astype('uint8'), thresh_val, levels, cv2.THRESH_BINARY)
            lines, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contour_sizes = [(cv2.contourArea(line), line) for line in lines]
            if contour_sizes == []:
                continue
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            cv2.drawContours(ct_copy[i,:,:], [biggest_contour], 0, (255), 1)
        ct_copy = np.squeeze(ct_copy)
        fig, axs = plt.subplots(1, 1)
        tracker = IndexTracker(axs, np.moveaxis(ct_copy,0,2))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()



if __name__ == '__main__':
    #------------------
    #options
    mode = 'sigmoid'
    num_slices = 500
    plot_masks = False
    plot_contours = True
    window_level = False
    window = 0.5 #.6
    level = 0.5 #.5
    #------------------
    
    print('Loading data...')

    # cts = np.load('./npydata/' + mode + '.npy')
    # contours = np.load('./npydata/' + mode + '_contours.npy')
    cts = np.load('./npydata/Spinal-Cord/cts/0.npy')
    contours = np.load('./npydata/Spinal-Cord/contours/0.npy')
    print(contours.shape)
    contours = contours#[:,0,:,:] #change structure
    cts = cts[0:num_slices]
    contours = contours[0:num_slices]
    if window_level:
        print('Changing window/level...')
        cts[np.where(cts > ((window/2) + level))] = ((window/2) + level)
        cts[np.where(cts < (level - (window/2)))] = (level - (window/2))
        if level - (window/2) <= 0:
            cts += np.abs(level - (window/2))
        else:
            cts -= np.abs(level - (window/2))
        cts *= 1./(window)
    print('Drawing contours...')
    display = display_contours(cts, contours)
    if plot_masks:
        display.plot_masks()
    if plot_contours:
        display.plot_contours()
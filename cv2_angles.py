import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt


file_numbers = ["06", "07", "09", "10", "12", "15", "16", "20", "21", "27"]
PATH_1st_part = "./input/img_00"
PATH_2nd_part = "_mask_Unet_toe.png"

def filename(number):
    return PATH_1st_part + number + PATH_2nd_part
    
# %%%
class CV2LineExtractor:
    line_tuple = collections.namedtuple("line", "rho theta")

    def __init__(self, file_name):
        self.filename = file_name
        self.image = cv2.imread(self.filename)
        self.main_lines = np.empty((2,), dtype=object)
        self.angle = 0

    def cv2_show_image(self):

        cv2.imshow('image', self.image)
        while True:
            k = cv2.waitKey(100) & 0xFF 

            if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == 27:
                print('ESC')
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        
    def sign_image(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap=plt.cm.gray)
        ax.set_title(str(self.angle) + r"$^\circ$")
        
        return fig, ax
    
    def save(self, folder_name):
        fig, ax = self.sign_image()
        self._save(fig, ax, folder_name)
        
    def _save(self, fig, ax, folder_name):
        fil = "./" + folder_name + self.filename.split("./input")[1]
        plt.savefig(fil)
        plt.close()

    def show(self, fig, ax):
        plt.show()

    def skeletonize(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        skeleton = cv2.ximgproc.thinning(gray)
        edges = cv2.Canny(skeleton, 50, 150, apertureSize = 3)

        minLineLength = 10
        maxLineGap = 1
        lines = cv2.HoughLines(edges, 1, np.pi/180, 10, minLineLength, maxLineGap)
        lines = np.vstack(lines)
        
        self.extract_main_lines(lines)

    def extract_main_lines(self, lines):
        rho, theta = lines[0]

        if rho < 0:
            rho *= -1
            theta -= np.pi

        self.main_lines[0] = CV2LineExtractor.line_tuple(rho=rho, theta=theta)
        self.main_lines[1] = self.second_main_line(lines)
        self.angle = np.round(180 - np.rad2deg(np.abs(self.main_lines[0].theta - self.main_lines[1].theta)), 2)



    def second_main_line(self, lines):
        for rho, theta in lines[1:len(lines)]:

            if rho < 0:
                rho *= -1
                theta -= np.pi

            is_close_rho = True if np.abs(rho-self.main_lines[0].rho) < 150 else False
            is_close_theta = True if np.abs(theta-self.main_lines[0].theta) < 0.25 else False

            if not is_close_rho and not is_close_theta:
                self.main_lines[1] = CV2LineExtractor.line_tuple(rho=rho, theta=theta)
                return self.main_lines[1]
        
        return self.main_lines[1]

    def draw_line(self, line):

        a = np.cos(line.theta)
        b = np.sin(line.theta)
        x0 = a*line.rho
        y0 = b*line.rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(self.image, (x1,y1), (x2,y2), (0,0,255), 2)

    def draw_lines(self):
        self.draw_line(self.main_lines[0])
        self.draw_line(self.main_lines[1])
        
    
    def run(self, show=False, cv2_show=False):
        self.skeletonize()
        self.draw_lines()
        
        if cv2_show:
            self.cv2_show_image()
        
        if show:
            fig, ax = self.sign_image()
            self.show(fig, ax)

        return self
# %%

if __name__ == "__main__":
    
    for file_number in file_numbers:
        f_name = filename(file_number)
        CV2LineExtractor(f_name).run().save(folder_name="homework_results_cv2")

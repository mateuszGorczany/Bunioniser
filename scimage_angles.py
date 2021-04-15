from skimage import io
from skimage.morphology import skeletonize, medial_axis
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

file_numbers = ["06", "07", "09", "10", "12", "15", "16", "20", "21", "27"]
PATH_1st_part = "./input/img_00"
PATH_2nd_part = "_mask_Unet_toe.png"


def filename(number):
    return PATH_1st_part + number + PATH_2nd_part


class LineExtractor:

    def __init__(self, file_name, method="lee"):
        self.image = io.imread(file_name)
        self.filename = file_name
        self.folder_name = "./homework_results"
        self.skeleton = 0
        self.angle = 0
        self.origin = np.array((0, self.image.shape[1]))  # Point on plane ( as a tuple )
        self.point_one = (0, 0)
        self.point_two = (0, 0)
        self.method = method

    @property
    def bool_image(self):
        image_gray = self.image[:, :, 0] / 255.0
        return skimage.img_as_bool(image_gray)

    def skeletonizer(self):
        image_bool = self.bool_image

        if self.method == "lee":
            self.skeleton = skeletonize(image_bool, method="lee")

        if self.method == "medial_axis":
            self.skeleton = medial_axis(image_bool)
            self.folder_name += "_" + self.method

    # Returns origin of image and Points of 2 main lines
    def extract_main_lines(self):

        self.skeletonizer()

        hough_space, angles, distances = hough_line(self.skeleton)
        hough_space, angles, distances = hough_line_peaks(hough_space,
                                                          angles,
                                                          distances,
                                                          num_peaks=2)

        self.angle = np.round(180 - np.rad2deg(np.abs(angles[0] - angles[1])), 2)
        self.point_one = (distances[0] - self.origin * np.cos(angles[0])) / np.sin(angles[0])
        self.point_two = (distances[1] - self.origin * np.cos(angles[1])) / np.sin(angles[1])


    def plot_result(self, save=False, filename=""):
        self.extract_main_lines()
        plt.imshow(self.image, cmap=plt.cm.gray)

        plt.plot(self.origin, self.point_one)
        plt.plot(self.origin, self.point_two)

        plt.xlim(self.origin)
        plt.ylim((self.image.shape[1], 0))
        plt.title(str(self.angle) + r"$^\circ$")

        if save:
            fil = self.folder_name + self.filename.split("./input")[1]
            plt.savefig(fil)
            plt.close()

        if not save:
            plt.show()


if __name__ == '__main__':
    for file_number in file_numbers:
        f_name = filename(file_number)
        LineExtractor(f_name, method="lee").plot_result(save=True)
        LineExtractor(f_name, method="medial_axis").plot_result(save=True)


import cv2
import numpy as np

kernel_size = (10, 10)  # output size
sigma = 2  # gaussian envelope
theta = 20  # real orientation
lambd = 0.5  # wavelength
gamma = np.pi * 0.5  # spatial aspect ratio
psi = 0  # phase
kernel = cv2.getGaborKernel(
    kernel_size, sigma, np.deg2rad(theta), lambd, gamma, psi, ktype=cv2.CV_32F
)
print(kernel)
sns.heatmap(kernel)
plt.show()

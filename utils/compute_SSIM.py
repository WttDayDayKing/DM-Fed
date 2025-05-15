import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim

### 计算两张图片的相似性即SSIM方法
def compute_SSIM():
    # 读取图像
    imageA = io.imread('/data/wtt/data_set/FL_CL/Matek-19/BAS/BAS_0001.tiff')  # 替换为您的图像路径
    imageB = io.imread('/data/wtt/data_set/FL_CL/Matek-19/BAS/BAS_0002.tiff')  # 替换为您的图像路径
    imageA=transform.resize(imageA, (224, 224), anti_aliasing=True)
    imageB=transform.resize(imageB,(224,224),anti_aliasing=True)
    # 将图像转换为灰度
    imageA_gray = np.dot(imageA[..., :3], [0.299, 0.587, 0.114])
    imageB_gray = np.dot(imageB[..., :3], [0.299, 0.587, 0.114])

    # 计算 SSIM
    similarity_index, diff = ssim(imageA_gray, imageB_gray, full=True)

    # 显示结果
    print(f'SSIM: {similarity_index}')

    # 显示差异图
    plt.imshow(diff, cmap='gray')
    plt.title('Difference Image')
    plt.colorbar()
    plt.show()


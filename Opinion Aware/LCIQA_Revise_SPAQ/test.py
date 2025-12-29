import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn
from skimage.util import view_as_blocks
from skimage import io, color

def rho_gen_gauss(temp):
    """计算 rho 特征的辅助函数"""
    return np.std(np.abs(temp)) / (np.mean(np.abs(temp)) + 1e-8)

def gama_gen_gauss(temp):
    """计算 gama 特征的辅助函数"""
    return np.mean(temp)  # 根据实际需求调整

def rho_dct(block):
    """计算 rho_dct 特征"""
    temp = dctn(block, norm='ortho')
    temp = temp.flatten()[1:]  # 排除 DC 成分
    return rho_gen_gauss(temp)

def gama_dct(block):
    """计算 gama_dct 特征"""
    temp = dctn(block, norm='ortho')
    temp = temp.flatten()[1:]  # 排除 DC 成分
    return gama_gen_gauss(temp)

def oriented1_dct_rho_config3(block):
    """计算 oriented1_dct_rho_config3 特征"""
    temp = dctn(block, norm='ortho')
    F = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    temp1 = temp[F != 0]
    std_gauss = np.std(np.abs(temp1))
    mean_abs = np.mean(np.abs(temp1))
    g1 = std_gauss / (mean_abs + 1e-8)
    return g1

def oriented2_dct_rho_config3(block):
    """计算 oriented2_dct_rho_config3 特征"""
    temp = dctn(block, norm='ortho')
    F = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    temp2 = temp[F != 0]
    std_gauss = np.std(np.abs(temp2))
    mean_abs = np.mean(np.abs(temp2))
    g2 = std_gauss / (mean_abs + 1e-8)
    return g2

def oriented3_dct_rho_config3(block):
    """计算 oriented3_dct_rho_config3 特征"""
    temp = dctn(block, norm='ortho')
    F = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ]).T
    temp3 = temp[F != 0]
    std_gauss = np.std(np.abs(temp3))
    mean_abs = np.mean(np.abs(temp3))
    g3 = std_gauss / (mean_abs + 1e-8)
    return g3

def dct_freq_bands(block):
    """计算 DCT 频带能量特征"""
    temp = dctn(block, norm='ortho')
    high_freq = temp[1:, 1:]  # 排除 DC 成分
    return np.sum(high_freq ** 2)

def pad_image(Img, block_size=(3, 3)):
    """
    填充图像，使其高度和宽度都能被块大小整除。

    参数:
    - Img: 输入的灰度图像，2D NumPy 数组。
    - block_size: 块的大小，默认为 (3, 3)。

    返回:
    - Img_padded: 填充后的图像。
    """
    pad_height = (block_size[0] - Img.shape[0] % block_size[0]) % block_size[0]
    pad_width = (block_size[1] - Img.shape[1] % block_size[1]) % block_size[1]
    return np.pad(Img, ((0, pad_height), (0, pad_width)), mode='reflect')

def extract_features_level(Img):
    """
    提取单层级的特征。

    参数:
    - Img: 灰度图像，2D NumPy 数组。

    返回:
    - features: 特征向量，NumPy 数组。
    """
    block_size = (3, 3)
    
    # 填充图像
    Img_padded = pad_image(Img, block_size)
    
    # 将图像划分为不重叠的块
    try:
        blocks = view_as_blocks(Img_padded, block_size)
    except ValueError as e:
        print(f"块划分失败: {e}")
        print(f"填充后的图像尺寸: {Img_padded.shape}")
        raise e

    num_blocks_x, num_blocks_y, _, _ = blocks.shape
    blocks = blocks.reshape(-1, block_size[0], block_size[1])

    # 计算各个特征
    coeff_freq_var = np.array([rho_dct(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)
    gama = np.array([gama_dct(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)
    ori1_rho = np.array([oriented1_dct_rho_config3(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)
    ori2_rho = np.array([oriented2_dct_rho_config3(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)
    ori3_rho = np.array([oriented3_dct_rho_config3(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)
    subband_energy = np.array([dct_freq_bands(block) for block in blocks]).reshape(num_blocks_x, num_blocks_y)

    # 计算 percentiles
    # coeff_freq_var: 降序排序
    rho_sorted = np.sort(coeff_freq_var.flatten())[::-1]
    rho_count = len(rho_sorted)
    percentile10_coeff_freq_var = rho_sorted[:int(np.ceil(rho_count / 10))].mean()
    percentile100_coeff_freq_var = rho_sorted.mean()

    # gama: 升序排序
    gama_sorted = np.sort(gama.flatten())
    gama_count = len(gama_sorted)
    percentile10_gama = gama_sorted[:int(np.ceil(gama_count / 10))].mean()
    percentile100_gama = gama_sorted.mean()

    # subband_energy: 降序排序
    subband_energy_sorted = np.sort(subband_energy.flatten())[::-1]
    subband_energy_count = len(subband_energy_sorted)
    percentile10_subband_energy = subband_energy_sorted[:int(np.ceil(subband_energy_count / 10))].mean()
    percentile100_subband_energy = subband_energy_sorted.mean()

    # 计算 ori_rho 的方差
    ori_rho = np.var(np.stack([ori1_rho, ori2_rho, ori3_rho], axis=-1), axis=-1)

    # orientation: 降序排序
    ori_sorted = np.sort(ori_rho.flatten())[::-1]
    ori_count = len(ori_sorted)
    percentile10_orientation = ori_sorted[:int(np.ceil(ori_count / 10))].mean()
    percentile100_orientation = ori_sorted.mean()

    # 合并特征
    features = np.array([
        percentile100_coeff_freq_var,
        percentile10_coeff_freq_var,
        percentile100_gama,
        percentile10_gama,
        percentile100_subband_energy,
        percentile10_subband_energy,
        percentile100_orientation,
        percentile10_orientation
    ])

    return features

def bliinds2_feature_extraction(I):
    """
    从图像中提取 BLIINDS-II 特征。

    参数:
    - I: 输入图像，NumPy 数组，形状为 (H, W, C) 或 (H, W)

    返回:
    - features: 特征向量，NumPy 数组
    """
    # 转换为灰度图像
    if I.ndim == 3:
        Img = I[:, :, 0].astype(np.float64)
    else:
        Img = I.astype(np.float64)

    # 定义高斯滤波参数
    sigma = 0.5  # 与 MATLAB 的 fspecial('gaussian',3) 相似

    # 第一级特征提取
    features_L1 = extract_features_level(Img)

    # 第二级特征提取
    Img1_filtered = gaussian_filter(Img, sigma=sigma)
    Img2 = Img1_filtered[1::2, 1::2]  # MATLAB 的 2:2:end 对应 Python 的 1::2
    features_L2 = extract_features_level(Img2)

    # 第三级特征提取
    Img2_filtered = gaussian_filter(Img2, sigma=sigma)
    Img3 = Img2_filtered[1::2, 1::2]
    features_L3 = extract_features_level(Img3)

    # 合并所有层的特征
    features = np.concatenate([features_L1, features_L2, features_L3])

    return features

if __name__ == "__main__":
    from skimage import io, color

    # 加载示例图像
    image_path = '/root/IQA_Dataset/KONIQ/1024x768/826373.jpg' # 替换为实际图像路径
    image = io.imread(image_path)
    if image.ndim == 3:
        image = color.rgb2gray(image)

    features = bliinds2_feature_extraction(image)
    print("提取的特征:", features)
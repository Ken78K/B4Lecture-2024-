"""主成分分析を行う."""

import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot2dscatter(data, xlabel, ylabel, title):
    """2次元の散布図を表示する.

    Args:
        data (np.ndarray): [x,y]の形で定義された2次元データ
        xlabel (str): x軸のラベル
        ylabel (str): y軸のラベル
        title (str): 散布図のタイトル
    """
    x_values = data[:, 0]
    y_values = data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_values, y_values)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.show()

def plot3dscatter(data, xlabel, ylabel, zlabel, title):
    """3次元の散布図を表示する.

    Args:
        data (np.ndarray): [x,y,z]の形で定義された3次元データ
        xlabel (str): x軸のラベル
        ylabel (str): y軸のラベル
        zlabel (str): z軸のラベル
        title (str): 散布図のタイトル
    """
    x_values = data[:, 0]
    y_values = data[:, 1]
    z_values = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(x_values, y_values, z_values)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    plt.show()

def pca(data):
    """主成分分析によって固有値を求める.

    Args:
        data (np.ndarray): 入力データ

    Returns:
        eige_nvals (np.ndarray): 固有値
        eigen_vecs (np.ndarray): 固有値ベクトル
        contributions (np.ndarray): 寄与率
    """
    # 標準化
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    standard_data = (data - mean_data) / std_data

    # 共分散行列を求める
    cov_data = np.cov(standard_data, rowvar = False)
    # 固有値、固有ベクトルを求める
    eigen_vals, eigen_vecs = np.linalg.eig(cov_data)

    # 固有値、固有ベクトルをソート
    sorted_indices = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    # 寄与率を求める
    sum = np.sum(eigen_vals, axis = 0)
    contributions = eigen_vals / sum

    return eigen_vals, eigen_vecs, contributions


def plot_line_2d(data, eigen_vecs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1])
    plt.axline(
        (0, 0),
        (eigen_vecs[0, 0], eigen_vecs[1, 0]),
        color = "red",
        label = "1st component"
    )
    ax.axline(
        (0, 0),
        (eigen_vecs[0, 1], eigen_vecs[1, 1]),
        color = "green",
        label = "2nd component"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Principal Components data1")
    ax.legend()
    plt.show()


def plot_line_3d(data, eigen_vecs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.axline(
        (0, 0),
        (eigen_vecs[0, 0], eigen_vecs[1, 0]),
        color = "red",
        label = "1st component"
    )
    ax.axline(
        (0, 0),
        (eigen_vecs[0, 1], eigen_vecs[1, 1]),
        color = "green",
        label = "2nd component"
    )
    ax.axline(
        (0, 0),
        (eigen_vecs[0,2], eigen_vecs[1,2]),
        color = "yellow",
        label = "3rd component"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Principal Components")
    ax.legend()
    plt.show()


def main():
    """入力ファイルの散布図を表示して主成分分析."""
    parser = argparse.ArgumentParser(description = "Process 3 file names.")

    parser.add_argument("file1", type = open, help = "First file name")
    parser.add_argument("file2", type = open, help = "Second file name")
    parser.add_argument("file3", type = open, help = "Third file name")

    args = parser.parse_args()

    data1 = np.loadtxt(args.file1, delimiter = ',')
    data2 = np.loadtxt(args.file2, delimiter = ',')
    data3 = np.loadtxt(args.file3, delimiter = ',')

    plot2dscatter(data1, "x", "y", "scatter of data1")
    plot3dscatter(data2, "x", "y", "z", "scatter of data2")

    eigen_vals1, eigen_vecs1, contributions1 = pca(data1)
    eigen_vals2, eigen_vecs2, contributions2 = pca(data2)
    eigen_vals3, eigen_vecs3, contributions3 = pca(data3)

    # print("data1's eigenvalues = ", eigen_vals1)
    # print("data2's eigenvalues = ", eigen_vals2)
    # print("data3's eigenvalues = ", eigen_vals3)
    # print("data1's contributions = ", contributions1)
    # print("data2's contributions = ", contributions2)

    plot_line_2d(data1, eigen_vecs1)
    # plot_line_3d(data2, eigen_vals2)

if __name__ == "__main__":
    main()

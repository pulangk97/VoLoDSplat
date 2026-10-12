import matplotlib.pyplot as plt
import numpy as np
def save_point_cloud_color(points, values=None, save_path='voxel_point_cloud.ply', max_value=1):
    if values is None:
        values = np.zeros((points.shape[0],))

    # 1. Normalize and Map Colors
    norm_values = np.clip(values / max_value, 0, 1)
    cmap = plt.get_cmap('jet')
    colors = (cmap(norm_values)[:, :3] * 255).astype(np.uint8)

    # 2. Writing the file
    try:
        with open(save_path, 'w') as f:
            # Header
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {points.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header"
            ]
            f.write('\n'.join(header) + '\n')

            # Data: Writing line by line
            for i in range(len(points)):
                f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

            f.flush() # Ensure all data is physically written to disk
        print(f"Successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving PLY: {e}")

def save_point_cloud_color_PC(points, colors=None, save_path='voxel_point_cloud.ply'):
    """
    Saves a point cloud with direct RGB color input.

    Args:
        points: (N, 3) numpy array (x, y, z)
        colors: (N, 3) numpy array (r, g, b). If None, defaults to white.
        save_path: Output filename.
    """
    if colors is None:
        # Default to white if no colors provided
        colors = np.ones((points.shape[0], 3)) * 255

    # 1. Handle Color Scaling
    # If colors are floats [0, 1], scale to [0, 255]. Otherwise, keep as is.
    if colors.dtype.kind == 'f':
        colors = (np.clip(colors, 0, 1) * 255)

    colors = colors.astype(np.uint8)

    # 2. Writing the file
    try:
        with open(save_path, 'w') as f:
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {points.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header"
            ]
            f.write('\n'.join(header) + '\n')

            # Efficient writing using string joining for large clouds
            for i in range(len(points)):
                f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

        print(f"Successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving PLY: {e}")


def save_point_cloud(points, values=None, save_path='point_cloud.ply'):
    if values is None:
        values = np.ones((points.shape[0],))

    with open(save_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float value\n')
        f.write('end_header\n')
        for p, v in zip(points, values):
            f.write(f'{p[0]} {p[1]} {p[2]} {v}\n')



import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def save_voxel_pc(coords, values, save_path, cmap='viridis'):
    """"""
    if coords.shape[0] == 0:
        return


    v_min, v_max = values.min(), values.max()
    norm_values = (values - v_min) / (v_max - v_min + 1e-8)


    cm = plt.get_cmap(cmap)
    colors = cm(norm_values.cpu().numpy())[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

# 点云平面拟合对照

从项目根目录运行。两种后端共用彩色候选区域、深度反投影、叉孔轮廓、净空检验和输出格式，差异在于前面板平面的初估方法。

1. 用已验证的 NumPy 后端检查三个观测：`python open3d_demo.py --backend numpy`。依次查看无遮挡、局部遮挡、相机横移后的候选状态、孔数和估计位姿。
2. 在支持 Open3D 的独立 Python 环境中安装本项目的基础依赖以及 `open3d`，然后运行 `python open3d_demo.py --backend open3d`。该路径把带单位的三维点传给 `PointCloud.segment_plane`，再通过两孔几何检查；输出含实际 Open3D 版本号。
3. 想查看保存的点云时，追加 `--show`。程序用 `o3d.io.read_point_cloud` 打开彩色 PLY，并显示世界坐标轴。默认文件是 `results/cloud/scene.ply`。如果需要重新生成，运行 `python cloud_demo.py`。也可用 `--ply 文件路径` 检查其他世界坐标系彩色点云。

PLY 用于点云浏览。检测仍然需要已配准的 RGB、深度与相机标定；本示例通过仿真器从同一组已知配置生成这些观测。当前交付环境已实测 NumPy 后端；Open3D 本机依赖未取得，运行指标待安装后补测。

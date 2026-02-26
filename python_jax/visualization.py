"""
可视化模块：绘制仿真结果
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def plot_tet_results(trajectory, X_ref, dt, skip_step=500):
    """绘制四面体单元仿真结果"""
    trajectory = np.asarray(trajectory)
    X_ref = np.asarray(X_ref)
    num_steps, n_nodes, _ = trajectory.shape
    time = np.arange(num_steps) * dt

    # 位移曲线
    plt.figure(figsize=(10, 4))
    disp_x = trajectory[:, 1, 0]
    plt.plot(time, disp_x, label="Node 1 (X-displacement)")
    plt.axhline(0, color="k", linestyle="--", alpha=0.3)
    plt.title(f"Node 1 Oscillation (Max Disp: {disp_x.max():.4f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement X")
    plt.grid(True)
    plt.legend()
    plt.savefig("displacement_curve.png")
    print("Displacement curve saved to 'displacement_curve.png'")
    plt.close()

    # 3D 动画
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames = trajectory[::skip_step]
    n_frames = len(frames)

    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-0.2, 1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Explicit Dynamics: Tet4 Oscillation")

    scat = ax.scatter([], [], [], c="b", s=50)
    scat_node1 = ax.scatter([], [], [], c="r", s=80, marker="*")
    edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    lines = [ax.plot([], [], [], "k-", alpha=0.5)[0] for _ in edges]

    def update(frame_idx):
        curr_X = X_ref + frames[frame_idx]
        current_time = frame_idx * skip_step * dt
        ax.set_title(f"Time: {current_time:.3f} s")
        scat._offsets3d = (curr_X[:, 0], curr_X[:, 1], curr_X[:, 2])
        scat_node1._offsets3d = ([curr_X[1, 0]], [curr_X[1, 1]], [curr_X[1, 2]])
        for line, edge in zip(lines, edges):
            p1, p2 = curr_X[edge[0]], curr_X[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        return [scat, scat_node1] + lines

    print(f"Generating animation with {n_frames} frames (skip_step={skip_step})...")
    ani = FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
    plt.show()


def plot_hex_results(trajectory, X_ref, dt, skip_step=200, conn=None):
    """绘制六面体单元仿真结果"""
    trajectory = np.asarray(trajectory)
    X_ref = np.asarray(X_ref)
    num_steps, n_nodes, _ = trajectory.shape
    time = np.arange(num_steps) * dt

    # 根据连接关系构建边：conn[0..7] 定义六面体节点顺序
    # 底面 0-1-2-3，顶面 4-5-6-7，柱边 0-4, 1-5, 2-6, 3-7
    if conn is not None:
        conn = np.asarray(conn)
        all_edges = []
        for e in range(conn.shape[0]):
            c = conn[e]
            all_edges.extend([
                [c[0], c[1]], [c[1], c[2]], [c[2], c[3]], [c[3], c[0]],  # 底面
                [c[4], c[5]], [c[5], c[6]], [c[6], c[7]], [c[7], c[4]],  # 顶面
                [c[0], c[4]], [c[1], c[5]], [c[2], c[6]], [c[3], c[7]]   # 柱边
            ])
        edges = all_edges
    else:
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 柱边
        ]

    # 自适应 skip_step，保证至少有 min_frames 帧动画
    min_frames = 50
    if num_steps <= skip_step or num_steps // skip_step < min_frames:
        skip_step = max(1, num_steps // min_frames)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames = trajectory[::skip_step]
    
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.5)
    ax.set_zlim(-0.2, 1.5)
    ax.set_title("C3D8R Hexahedral Element - Total Lagrangian explicit")

    scat = ax.scatter([], [], [], c="b", s=30)
    lines = [ax.plot([], [], [], "k-", alpha=0.6)[0] for _ in edges]

    def update(frame_idx):
        curr_X = X_ref + frames[frame_idx]
        ax.set_title(f"Time: {frame_idx * skip_step * dt:.4f} s")
        scat._offsets3d = (curr_X[:, 0], curr_X[:, 1], curr_X[:, 2])
        for line, edge in zip(lines, edges):
            p1, p2 = curr_X[edge[0]], curr_X[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        return [scat] + lines

    print(f"Generating animation... (n_frames={len(frames)}, skip_step={skip_step})")
    ani = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=False)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

FOV = 50
CAMERA_HEIGHT = 0.12
PLOT_STEPS = -1

# Read ground truths
gt_pos = open(f"gt_positions-{FOV}.csv").readlines()
gt_pos = np.array(list(map(lambda l: list(map(float, l.split(','))), gt_pos)))

gt_pos[:, 0] *= -1

gt_rot = open(f"gt_orientations-{FOV}.csv").readlines()
gt_rot = list(map(lambda l: list(map(float, l.split(','))), gt_rot))
gt_rot = np.array(list(map(lambda r: [r[0:3], r[3:6], r[6:9]], gt_rot)))

# Read VO results
vo_pos = [gt_pos[0] / CAMERA_HEIGHT]

lines = open(f"out-{FOV}.txt").readlines()
T_acc = np.identity(4)
T_acc[:3, :3] = gt_rot[0]
T_acc[:3, 3] = vo_pos[0]
for [pose, matches, inliers] in zip(lines[::3], lines[1::3], lines[2::3]):
    pose = list(map(float, pose.split()))

    T = np.array([pose[0:4], pose[4:8], pose[8:12], [0.0, 0.0, 0.0, 1.0]])
    T_acc = T @ T_acc
    pos = T_acc[:3, 3]
    rot = T_acc[:3, :3]

    vo_pos.append(pos)
vo_pos = np.array(vo_pos) * CAMERA_HEIGHT

plt.plot(gt_pos[:PLOT_STEPS,0], gt_pos[:PLOT_STEPS,2])
plt.plot(vo_pos[:PLOT_STEPS,0], vo_pos[:PLOT_STEPS,2])
plt.axis('square')
plt.show()
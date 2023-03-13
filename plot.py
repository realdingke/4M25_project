import os
import numpy as np
import matplotlib.pyplot as plt

FOV = 50
CAMERA_HEIGHT = 0.12
PLOT_STEPS = -1
FLOOR_SIZE = (2, 2)
CENTER = (0, 0)

# Read ground truths
gt_pos = open(f"imgs/gt_positions-{FOV}.csv").readlines()
gt_pos = np.array(list(map(lambda l: list(map(float, l.split(','))), gt_pos)))

gt_pos[:, 0] *= -1

gt_rot = open(f"imgs/gt_orientations-{FOV}.csv").readlines()
gt_rot = list(map(lambda l: list(map(float, l.split(','))), gt_rot))
gt_rot = np.array(list(map(lambda r: [r[0:3], r[3:6], r[6:9]], gt_rot)))

# Read VO results
vo_pos = [gt_pos[0] / CAMERA_HEIGHT]
vo_rot = [gt_rot[0]]

vo_match = []
vo_inliers = []

lines = open(f"imgs/out-{FOV}.txt").readlines()
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
    vo_rot.append(rot)
    vo_match.append(float(matches))
    vo_inliers.append(float(inliers))

vo_pos = np.array(vo_pos) * CAMERA_HEIGHT
vo_match = np.array(vo_match)
vo_inliers = np.array(vo_inliers)

x = vo_match
print("Match:")
print(np.quantile(x, 0.9), np.quantile(x, 0.5), np.quantile(x, 0.1))

x = vo_inliers
print("Inliers:")
print(np.quantile(x, 0.9), np.quantile(x, 0.5), np.quantile(x, 0.1))


plt.figure(figsize = (8,6))
###no tick
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)

plt.scatter(CENTER[0], CENTER[1], s=5, c='r')
plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] + FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] - FLOOR_SIZE[1]/2), c='k')
plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] - FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
plt.plot((CENTER[0] + FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
plt.plot(gt_pos[:PLOT_STEPS,0], gt_pos[:PLOT_STEPS,2], c='g', label="Ground Truth")
plt.plot(vo_pos[:PLOT_STEPS,0], vo_pos[:PLOT_STEPS,2], c='b', label="Visual Odometry")
plt.axis('square')
plt.xlim(-(FLOOR_SIZE[0]/2 + 0.5), FLOOR_SIZE[0]/2 + 0.5)
plt.ylim(-(FLOOR_SIZE[1]/2 + 0.5), FLOOR_SIZE[1]/2 + 0.5)
plt.legend()

root_dir = os.getcwd()
plt.savefig(os.path.join(root_dir, "VO-groundtruth trajectory plot.png"), bbox_inches='tight', pad_inches=0)

plt.show()

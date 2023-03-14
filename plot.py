import os
import numpy as np
from scipy import spatial
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

FOV = 150
PLOT_STEPS = -1
FLOOR_SIZE = (2, 2)
CENTER = (0, 0)



# Read ground truths
def parsePosition(line):
    vals = list(map(float, line.split(',')))
    return [-vals[0], vals[2]]

gt_pos = open(f"imgs/gt_positions--{FOV}.csv").readlines()
gt_pos = np.array(list(map(parsePosition, gt_pos)))

def parseRotation(line):
    vals = list(map(float, line.split(',')))
    mat = spatial.transform.Rotation.from_matrix([vals[0:3], vals[3:6], vals[6:9]])
    return -mat.as_euler("xyz")[1]

gt_rot = open(f"imgs/gt_orientations--{FOV}.csv").readlines()
gt_rot = np.array(list(map(parseRotation, gt_rot)))



# Read VO results
vo_pos = [[0.0, 0.0]]
vo_rot = [0.0]
vo_match = []
vo_inliers = []
T_acc = np.identity(4)

lines = open(f"imgs/out-{FOV}.txt").readlines()
for [pose, matches, inliers] in zip(lines[::3], lines[1::3], lines[2::3]):
    pose = list(map(float, pose.split()))

    T = np.array([pose[0:4], pose[4:8], pose[8:12], [0.0, 0.0, 0.0, 1.0]])
    T_acc = T @ T_acc
    pos = T_acc[:3, 3]
    rot = spatial.transform.Rotation.from_matrix(T_acc[:3, :3])

    vo_pos.append([pos[0], pos[2]])
    vo_rot.append(rot.as_euler("xyz")[2])
    vo_match.append(float(matches))
    vo_inliers.append(float(inliers))

vo_pos = np.array(vo_pos)
vo_rot = np.array(vo_rot)
vo_match = np.array(vo_match)
vo_inliers = np.array(vo_inliers)



# Truncate longer (in case of old images in folder / partial data)
if len(vo_pos) < len(gt_pos):
    gt_pos = gt_pos[:len(vo_pos)]
    gt_rot = gt_rot[:len(vo_pos)]

if len(gt_pos) < len(vo_pos):
    vo_pos = vo_pos[:len(gt_pos)]
    vo_rot = vo_rot[:len(gt_pos)]



# Align with ground truth
theta = gt_rot[0]
vo_rot += theta
theta -= np.pi / 2
T = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
vo_pos = (T @ vo_pos.T).T



# Calculate dposes
gt_dpos = gt_pos[:-1] - gt_pos[1:]
gt_drot = gt_rot[:-1] - gt_rot[1:]
vo_dpos = vo_pos[:-1] - vo_pos[1:]
vo_drot = vo_rot[:-1] - vo_rot[1:]



# Solve for scale
huber = HuberRegressor().fit(vo_dpos.reshape((-1, 1)), gt_dpos.ravel())
scale = huber.coef_
vo_dpos *= scale
vo_pos = (vo_pos * scale) + gt_pos[0]



# Compute error [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite]
err_rot = np.atan2(np.sin(vo_drot-gt_drot), np.cos(vo_drot-gt_drot)) # Angular differences mapped to [-pi, pi]
E_rot = np.mean(err_rot)
err_trans = vo_dpos - gt_dpos
err_trans = np.sqrt(err_trans[:,0]**2 + err_trans[:,1]**2)
E_trans = np.mean(err_trans)

# ------
# Output
# ------

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
plt.plot(gt_pos[:PLOT_STEPS,0], gt_pos[:PLOT_STEPS,1], c='g', label="Ground Truth")
plt.plot(vo_pos[:PLOT_STEPS,0], vo_pos[:PLOT_STEPS,1], c='b', label="Visual Odometry")
plt.axis('square')
plt.xlim(-(FLOOR_SIZE[0]/2 + 0.5), FLOOR_SIZE[0]/2 + 0.5)
plt.ylim(-(FLOOR_SIZE[1]/2 + 0.5), FLOOR_SIZE[1]/2 + 0.5)
plt.legend()

root_dir = os.getcwd()
plt.savefig(os.path.join(root_dir, "VO-groundtruth trajectory plot.png"), bbox_inches='tight', pad_inches=0)

plt.show()

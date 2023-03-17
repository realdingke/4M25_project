import os
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

WORLDS = ["world_downwardcam.wbt", "hall.wbt", "city.wbt", "complete_apartment.wbt", "village_realistic.wbt"]
FOVS = [40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220]

PLOT_STEPS = -1
FLOOR_SIZE = (2, 2)
CENTER = (0, 0)

SMOOTH_STEPS = 20


def readDataSeries(world, fov):
    # Read ground truths
    def parsePosition(line):
        vals = list(map(float, line.split(',')))
        return [-vals[0], vals[2]]

    with open(f"imgs/gt_positions-{world}-{fov}.csv") as f:
        gt_pos = f.readlines()
    gt_pos = np.array(list(map(parsePosition, gt_pos)))

    def parseRotation(line):
        vals = list(map(float, line.split(',')))
        mat = spatial.transform.Rotation.from_matrix([vals[0:3], vals[3:6], vals[6:9]])
        return -mat.as_euler("xyz")[1]

    with open(f"imgs/gt_orientations-{world}-{fov}.csv") as f:
        gt_rot = f.readlines()
    gt_rot = np.array(list(map(parseRotation, gt_rot)))



    # Read VO results
    vo_pos = [[0.0, 0.0]]
    vo_rot = [0.0]
    vo_match = []
    vo_inliers = []
    vo_dist = []
    T_acc = np.identity(4)

    with open(f"imgs/out-{world}-{fov}.txt") as f:
        lines = f.readlines()
    for [pose, matches, inliers, match_dist] in zip(lines[::4], lines[1::4], lines[2::4], lines[3::4]):
        pose = list(map(float, pose.split()))

        T = np.array([pose[0:4], pose[4:8], pose[8:12], [0.0, 0.0, 0.0, 1.0]])
        T_acc = T @ T_acc
        pos = T_acc[:3, 3]
        rot = spatial.transform.Rotation.from_matrix(T_acc[:3, :3])

        vo_pos.append([pos[0], pos[2]])
        vo_rot.append(rot.as_euler("xyz")[2])
        vo_match.append(float(matches))
        vo_inliers.append(float(inliers))
        vo_dist.append(float(match_dist) / float(inliers))

    vo_pos = np.array(vo_pos)
    vo_rot = np.array(vo_rot)
    vo_match = np.array(vo_match)
    vo_inliers = np.array(vo_inliers)
    vo_dist = np.array(vo_dist)



    # Truncate longer (in case of old images in folder / partial data)
    error = False
    if len(vo_pos) < len(gt_pos):
        gt_pos = gt_pos[:len(vo_pos)]
        gt_rot = gt_rot[:len(vo_pos)]
        error = True

    if len(gt_pos) < len(vo_pos):
        vo_pos = vo_pos[:len(gt_pos)]
        vo_rot = vo_rot[:len(gt_pos)]
        error = True



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
    gt_drot = gt_rot[:-1] - gt_rot[1:]
    gt_dposr = gt_pos[:-1] - gt_pos[1:]
    gt_dpos = np.zeros_like(gt_dposr)
    gt_dpos[:,0] = np.cos(gt_rot[:-1]) * gt_dposr[:,0] + np.sin(gt_rot[:-1]) * gt_dposr[:,1]
    gt_dpos[:,1] = np.cos(gt_rot[:-1]) * gt_dposr[:,1] - np.sin(gt_rot[:-1]) * gt_dposr[:,0]

    vo_drot = vo_rot[:-1] - vo_rot[1:]
    vo_dposr = vo_pos[:-1] - vo_pos[1:]
    vo_dpos = np.zeros_like(vo_dposr)
    vo_dpos[:,0] = np.cos(vo_rot[:-1]) * vo_dposr[:,0] + np.sin(vo_rot[:-1]) * vo_dposr[:,1]
    vo_dpos[:,1] = np.cos(vo_rot[:-1]) * vo_dposr[:,1] - np.sin(vo_rot[:-1]) * vo_dposr[:,0]

    gt_pos = np.cumsum(gt_dpos, axis=0)
    vo_pos = np.cumsum(vo_dpos, axis=0)

    gt_pos = np.insert(gt_pos, 0, [0.0, 0.0], 0)
    vo_pos = np.insert(vo_pos, 0, [0.0, 0.0], 0)


    # Solve for scale
    if len(vo_dpos):
        huber = HuberRegressor().fit(vo_dpos.reshape((-1, 1)), gt_dpos.ravel())
        scale = huber.coef_
    else:
        scale = 1 / 0.012 # Cam-height approximation
    vo_dpos *= scale
    vo_pos = (vo_pos * scale) + gt_pos[0]



    # Compute error [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite]
    err_rot = np.abs(np.arctan2(np.sin(vo_drot-gt_drot), np.cos(vo_drot-gt_drot))) # Angular differences mapped to [-pi, pi]
    err_trans = vo_dpos - gt_dpos
    err_trans = np.sqrt(err_trans[:,0]**2 + err_trans[:,1]**2)

    return pd.DataFrame({"vo_match":vo_match, "vo_inliers":vo_inliers, "gt_dposx":gt_dpos[:,0],
            "gt_dposy":gt_dpos[:,1], "gt_drot":gt_drot, "vo_dposx":vo_dpos[:,0], "vo_dposy":vo_dpos[:,1],
            "vo_drot":vo_drot, "err_rot":err_rot, "err_trans":err_trans, "error":error}), \
        pd.DataFrame({"gt_posx":gt_pos[:,0], "gt_posy":gt_pos[:,1], "gt_rot":gt_rot,
            "vo_posx":vo_pos[:,0], "vo_posy":vo_pos[:,1], "vo_rot":vo_rot, "error":error})



# Read all results
data = {}
for world in WORLDS:
    worlddata = {}
    for fov in FOVS:
        dfdel, dfabs = readDataSeries(world, fov)
        if dfabs["error"].any(): continue
        worlddata[fov] = dfdel
    data[world] = worlddata

# Normalise rot/trans errors
cum_rot = 0.0
cum_trans = 0.0
for world in data:
    for fov in data[world]:

        cum_rot += np.sum(data[world][fov]["err_rot"])
        cum_trans += np.sum(data[world][fov]["err_trans"])

k = cum_trans / cum_rot
# Apply normalisation
for world in data:
    for fov in data[world]:
        data[world][fov]["err_rot"] *= k


for world in data:
    best = 999999999999999
    best_fov = 0
    worst = 0
    worst_fov = 0

    fov_errs = []
    for fov in data[world]:
        E = data[world][fov]["err_rot"] + data[world][fov]["err_trans"]

        Etot = np.sum(E)
        if Etot < best:
            best = Etot
            best_fov = fov
        if Etot > worst:
            worst = Etot
            worst_fov = fov

        Ebuckets = np.array(list(map(np.sum, np.split(E, SMOOTH_STEPS))))
        fov_errs.append(Ebuckets)

    fov_errs = np.vstack(fov_errs)
    theory = np.amin(fov_errs, axis=0)
    theory = np.sum(theory)


        

    print("===============")
    print(world)
    print("Best @", best_fov, "degrees:", best)
    print("Worst @", worst_fov, "degrees:", worst)
    print("Theoretical Best:", theory)

combined = {}
for world in data:
    for fov in data[world]:
        if not fov in combined:
            combined[fov] = []
        combined[fov].append(data[world][fov])


mtch = []
mtch10 = []
inlier = []
inlier10 = []  
for fov in FOVS:
    df = pd.concat(combined[fov])
    mtch.append(np.quantile(df["vo_match"], 0.5))
    mtch10.append(np.quantile(df["vo_match"], 0.1))
    inlier.append(np.quantile(df["vo_inliers"], 0.5))
    inlier10.append(np.quantile(df["vo_inliers"], 0.1))

plt.plot(FOVS, mtch, c='b', label="Ave. Matches")
plt.plot(FOVS, mtch10, c='0', label="10% Low Matches")
plt.legend()
plt.show()

plt.plot(FOVS, inlier, c='b', label="Ave. Inliers")
plt.plot(FOVS, inlier10, c='0', label="10% Low Inliers")
plt.legend()
plt.show()

# ------
# Plot a trajectory
# ------

dfdel, dfabs = readDataSeries("world_downwardcam.wbt", 160)


plt.figure(figsize = (8,6))
###no tick
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)

plt.scatter(CENTER[0], CENTER[1], s=5, c='r')
# plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] + FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
# plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] - FLOOR_SIZE[1]/2), c='k')
# plt.plot((CENTER[0] - FLOOR_SIZE[0]/2, CENTER[0] - FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
# plt.plot((CENTER[0] + FLOOR_SIZE[0]/2, CENTER[0] + FLOOR_SIZE[0]/2), (CENTER[1] - FLOOR_SIZE[1]/2, CENTER[1] + FLOOR_SIZE[1]/2), c='k')
#plt.plot(dfabs["gt_posx"], dfabs["gt_posy"], c='g', label="Ground Truth")
plt.plot(dfabs["vo_posx"], dfabs["vo_posy"], c='b', label="Visual Odometry")
# plt.axis('square')
# plt.xlim(-(FLOOR_SIZE[0]/2 + 0.5), FLOOR_SIZE[0]/2 + 0.5)
# plt.ylim(-(FLOOR_SIZE[1]/2 + 0.5), FLOOR_SIZE[1]/2 + 0.5)
plt.legend()

root_dir = os.getcwd()
plt.savefig(os.path.join(root_dir, "VO-groundtruth trajectory plot.png"), bbox_inches='tight', pad_inches=0)

plt.show()

"""mapping controller."""

from controller import Robot, Supervisor, Camera, CameraRecognitionObject, InertialUnit, DistanceSensor, PositionSensor, Keyboard
# import math
import os
import pickle
import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

TIME_STEP = 32

# create the Robot instance.
# robot = Robot()

# get the time step of the current world.
# timestep = int(robot.getBasicTimeStep())

# #enable distance sensors
# fds = robot.getDevice('front_ds')
# lds = robot.getDevice('left_ds')
# rds = robot.getDevice('right_ds')
# fds.enable(timestep)
# lds.enable(timestep)
# rds.enable(timestep)

# # getting the position sensors
# lps = robot.getDevice('left wheel sensor')
# rps = robot.getDevice('right wheel sensor')
# lps.enable(timestep)
# rps.enable(timestep)

# enable camera and recognition
# camera = robot.getDevice('camera1')
# camera.enable(timestep)

# #enable imu
# imu = robot.getDevice('inertial unit')
# imu.enable(timestep)

# # values for robot
# wheel_radius = 1.6 / 2.0
# wheel_circ = 2 * 3.14 * wheel_radius
# enc_unit = wheel_circ / 6.28
max_speed = 4
# # distance between wheels in inches and middle of the robot d_mid
# d = 2.28
# d_mid = d / 2.0

# # STARTING VARIABLES FOR CONFIGURATION
# # state of robot
# dir = "North"
# # new position to keep track of x,y as the robot keeps moving
# new_pos = [15.0, -15.0]


# def print_measurements():
#     print(f"left: {pos_s_to_inches(lps.getValue())}, right: {pos_s_to_inches(rps.getValue())}, imu: {(imu.getRollPitchYaw()[2] * 180) / 3.14159}")


# def get_p_sensors_vals():
#     # returns left and right position sensors
#     return pos_s_to_inches(lps.getValue()), pos_s_to_inches(rps.getValue())


# def get_d_sensors_vals():
#     # returns left, front, right sensors in inches
#     return m_to_i(lds.getValue()), m_to_i(fds.getValue()), m_to_i(rds.getValue())


# def get_time(distance, speed):
#     return distance / speed


# def stop_motors():
#     leftMotor.setVelocity(0)
#     rightMotor.setVelocity(0)
#     #print("Motors stopped.")


# def get_rot_speed_rad(degrees, seconds, wheel_radius, d_mid):
#     circle = d_mid * 2 * math.pi
#     dist = (degrees / 360) * circle
#     linear_vel = dist / seconds
#     left_wheel_speed = linear_vel / wheel_radius
#     right_wheel_speed = -1 * linear_vel / wheel_radius
#     return left_wheel_speed, right_wheel_speed


# def rotate(degrees, seconds, timestep, direction):
#     global last_vals
#     # get the left and right rotaional speeds to turn x degrees in y seconds
#     left, right = get_rot_speed_rad(degrees, seconds, wheel_radius, d_mid)
#     end_time = seconds + robot.getTime()
#     while robot.step(timestep) != -1:
#         # update and print the robot's details
#         update_robot(rotating=True)
#         # still update the last vals
#         vals = get_p_sensors_vals()
#         last_vals = vals
#         print(f"Rotating {direction}...")
#         if robot.getTime() < end_time:
#             leftMotor.setVelocity(left)
#             rightMotor.setVelocity(right)
#         else:
#             stop_motors()
#             break


# def turn_left(ts, degrees=-90.5):
#     rotate(degrees, 1.5, ts, "left")


# def turn_right(ts, degrees=90.5):
#     rotate(degrees, 1.5, ts, "right")


# def get_robot_x_y(vals):
#     global new_pos
#     global last_vals
#     diff = [vals[0] - last_vals[0], vals[1] - last_vals[1]]
#     for i in range(len(diff)):
#         diff[i] = math.fabs(diff[i])

#     if math.fabs(diff[0]) >= .3 or math.fabs(diff[1]) >= .3:
#             diff[0] = 0.3
#             diff[1] = 0.3

#     # diff average of the left and right wheel
#     diff_avg = (diff[0]+ diff[1]) / 2.0

#     # x and y are dependent on the direction the robot is moving in
#     if dir == "North":
#         x = new_pos[0]
#         y = new_pos[1] + diff_avg
#     elif dir == "West":
#         x = new_pos[0] - diff_avg
#         y = new_pos[1]
#     elif dir == "East":
#         x = new_pos[0] + diff_avg
#         y = new_pos[1]
#     elif dir == "South":
#         x = new_pos[0]
#         y = new_pos[1] - diff_avg
#     # update the last vals
#     last_vals = vals
#     # store the new x and y into the new_pos
#     new_pos = x, y
#     return x, y


# def get_direction(imu_val):
#     if (imu_val <= -135 and imu_val >= -180) or (135 <= imu_val <= 180):
#         dir = "North"
#     elif imu_val <= -45 and imu_val > -135:
#         dir = "West"
#     elif 45 <= imu_val <= 135:
#         dir = "East"
#     elif (-45 < imu_val <= 0) or (0 <= imu_val < 45):
#         dir = "South"
#     return dir


# def get_available_turns(walls):
#     # a 0 for a wall means there is no wall in that direction
#     # available_turns also returned in order [left, front, right]
#     # 0, means not available, 1 means available
#     # count is the number of available turns
#     count = 0
#     available = [0, 0, 0]
#     for i in range(len(walls)):
#         if walls[i] == 0:
#             available[i] = 1
#             count += 1
#         # otherwise if a wall is there, walls[i] == 1, leave it as a 0
#     return available, count


# def face_north(ts):
#     global dir
#     while robot.step(ts) != -1:
#         if dir == "West":
#             # turn right to face north again
#             turn_right(ts)
#         elif dir == "South":
#             # turn left twice to face north again
#             turn_left(ts)
#             turn_left(ts)
#         elif dir == "East":
#             # turn left to face north again
#             turn_left(ts)
#         if dir == "North":
#             break
#     dir = "North"



class BraitenbergController(object):
    def __init__(self, lds, rds, lm, rm):
        rds.enable(TIME_STEP)
        lds.enable(TIME_STEP)
        self.lds = lds
        self.rds = rds

        lm.setPosition(float('inf'))
        lm.setVelocity(0.0)
        rm.setPosition(float('inf'))
        rm.setVelocity(0.0)
        self.lm = lm
        self.rm = rm

        self.WeightMatrix = np.array([0, 0.1, 0.1, 0]).reshape(2, 2)

    def update(self, step):
        # read sensors
        left_dist = self.lds.getValue()
        right_dist = self.rds.getValue()
        dist_vec = np.array([left_dist, right_dist]).reshape(2, 1)

        angular_velocities = self.WeightMatrix @ dist_vec

        # actuate wheel motors
        self.lm.setVelocity(angular_velocities[0])
        self.rm.setVelocity(angular_velocities[1])



class RuleBasedController(object):
    def __init__(self, lds, rds, lm, rm):
        rds.enable(TIME_STEP)
        lds.enable(TIME_STEP)
        self.lds = lds
        self.rds = rds

        lm.setPosition(float('inf'))
        lm.setVelocity(0.0)
        rm.setPosition(float('inf'))
        rm.setVelocity(0.0)
        self.lm = lm
        self.rm = rm

    def update(self, step):
        # read sensors
        left_dist = self.lds.getValue()
        right_dist = self.rds.getValue()

        # compute behavior (user functions)
        left = self.compute_left_speed(left_dist, right_dist)
        right = self.compute_right_speed(left_dist, right_dist)

        # actuate wheel motors
        self.lm.setVelocity(left)
        self.rm.setVelocity(right)

    def compute_left_speed(self, l_dist, r_dist):
        if l_dist < r_dist:
            return max_speed*0.45
        else:
            return max_speed*0.5

    def compute_right_speed(self, l_dist, r_dist):
        if l_dist < r_dist:
            return max_speed*0.5
        else:
            return max_speed*0.45
        


class KeyboardController(object):
    def __init__(self, kb, lm, rm, vel=6.0, turn=0.5):
        self.kb = kb

        lm.setPosition(float('inf'))
        lm.setVelocity(0.0)
        rm.setPosition(float('inf'))
        rm.setVelocity(0.0)
        self.lm = lm
        self.rm = rm

        self.cmd = {
            ord('W'): (vel, vel),
            ord('A'): (-turn, turn),
            ord('D'): (turn, -turn),
            ord('S'): (-vel, -vel),
            ord('Q'): (0, 0)
        }

    def update(self, step):
        key = self.kb.getKey()
        if key in self.cmd:
            left, right = self.cmd[key]
            self.lm.setVelocity(left)
            self.rm.setVelocity(right)



class ReplayController(object):
    def __init__(self, lm, rm, savefile):
        self.lm = lm
        self.rm = rm
        lm.setPosition(float('inf'))
        lm.setVelocity(0.0)
        rm.setPosition(float('inf'))
        rm.setVelocity(0.0)
        self.savefile = open(savefile, "r")
        
        self.getcmd()
        
    def getcmd(self):
        line = self.savefile.readline().split(' ')
        if line == [""]:
            self.nextstep = 9999999999
            return
        self.nextstep = int(line[0])
        self.nextl = float(line[1])
        self.nextr = float(line[2])
        
    def update(self, step):
        if step != self.nextstep: return
        
        self.lm.setVelocity(self.nextl)
        self.rm.setVelocity(self.nextr)
        
        self.getcmd()
    
class ReplayRecorder(object):
    def __init__(self, lm, rm, savefile):
        self.lm = lm
        self.rm = rm
        self.savefile = open(savefile, "w")
        
        self.lastl = None
        self.lastr = None
        
    def update(self, step):
        curl = self.lm.getVelocity()
        curr = self.rm.getVelocity()
        
        if curl == self.lastl \
            and curr == self.lastr:
            return
            
        self.lastl = curl
        self.lastr = curr
        
        self.savefile.write(f"{step} {curl} {curr}\n")
        
    def close(self):
        self.savefile.close()



def execute(supervisor, world, fov, maxsteps=2000):

    robot = supervisor
    robot_node = supervisor.getFromDef('epuck')

    keyboard = Keyboard()
    keyboard.enable(TIME_STEP)

    camera = robot.getDevice('camera1')
    
    # Set FOV
    camera.setFov(fov / 180 * np.pi)
    camera.enable(TIME_STEP)

    # left_sensor = robot.getDevice("left_sensor")
    # right_sensor = robot.getDevice("right_sensor")
    # left_sensor.enable(TIME_STEP)
    # right_sensor.enable(TIME_STEP)
    lds = robot.getDevice('left_ds')
    rds = robot.getDevice('right_ds')

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')

    #CONTROLLER = RuleBasedController(lds, rds, left_motor, right_motor)
    #CONTROLLER = KeyboardController(keyboard, left_motor, right_motor)
    CONTROLLER = ReplayController(left_motor, right_motor, f"{world}.txt")
    
    RECORDER = None
    #RECORDER = ReplayRecorder(left_motor, right_motor, f"hall.wbt.txt")

    root_dir = os.getcwd()
    img_dir = os.path.join(root_dir, "imgs")

    gt_position_array = []
    gt_orientation_array = []

    ### get camera intrinsic matrix
    K = np.array([[camera.getFocalDistance(), 0.0, camera.getWidth()/2],
        [0.0, camera.getFocalDistance(), camera.getHeight()/2],
        [0.0, 0.0, 1.0]])
    print("camera intrinsic matrix is:", K)

    StepCounter = 0
    while robot.step(TIME_STEP) != -1:

        CONTROLLER.update(StepCounter)
        
        if RECORDER != None:
            RECORDER.update(StepCounter)

        #read camera image and display it
        # img = camera.getImageArray()
        # img_l = cv.imread(img_l, cv.IMREAD_GRAYSCALE)
        # img_r = cv.imread(img_r, cv.IMREAD_GRAYSCALE)
        # # img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        # # img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        # stereo = np.concatenate((img_l, img_r), axis=1)
        # cv.imshow("forward camera image", np.array(img))

        #get ground truth position and orientation from supervisor
        gt_position = robot_node.getPosition()
        gt_orientation = robot_node.getOrientation()

        if StepCounter%20 == 0:
            print(f"current robot position is at {gt_position}; current robot orientation is {gt_orientation}")
            gt_position_array.append(np.array(gt_position).flatten())
            gt_orientation_array.append(np.array(gt_orientation).flatten())

            # cv.imwrite(img_dir + f"Image No.{StepCounter//20}.png", img)
            # plt.imshow(img, interpolation='nearest')
            # plt.axis('off')
            # plt.savefig(os.path.join(img_dir, f"image-{world}-{fov}-{StepCounter//20}.png"), bbox_inches='tight', pad_inches=0)

            camera.saveImage(os.path.join(img_dir, f"image-{world}-{fov}-{StepCounter//20}.png"), -1)

            print(f"Image No.{StepCounter//20} at step {StepCounter} saved")

        if StepCounter > maxsteps:
            break

        StepCounter += 1

    np.savetxt(os.path.join(img_dir, f"gt_positions-{world}-{fov}.csv"), gt_position_array, delimiter=",", fmt='%s')
    np.savetxt(os.path.join(img_dir, f"gt_orientations-{world}-{fov}.csv"), gt_orientation_array, delimiter=",", fmt='%s')
    print("Ground truth data saved")
    
    if RECORDER != None:
        RECORDER.close()
        print("Saved path")

def main():
    run_all = True
    worlds = ["world_downwardcam.wbt", "hall.wbt", "city.wbt", "complete_apartment.wbt", "village_realistic.wbt"]
    fovs = [40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220]

    maxsteps = 4000

    os.chdir("../..")
    localpath = lambda f: os.path.join(os.getcwd()+"/worlds", f)

    supervisor = Supervisor()  # create Supervisor instance

    if not run_all:
        execute(supervisor, "", fovs[0], maxsteps=maxsteps)
        return
    
    try:
        state_file = open(localpath('state.p'), 'rb')
        state = pickle.load(state_file)
        state_file.close()
    except FileNotFoundError:
        # Set initial configuration
        pickle.dump([0, 0], open(localpath('state.p'), 'wb'))
        supervisor.worldLoad(localpath(worlds[0]))

    execute(supervisor, worlds[state[0]], fovs[state[1]], maxsteps=maxsteps)

    if state[0] == len(worlds) - 1 \
        and state[1] == len(fovs) - 1:
        print("Finished gathering data")
        os.remove(localpath('state.p'))
        supervisor.simulationQuit(0)
        return
    
    state[1] += 1
    if state[1] == len(fovs):
        state[0] += 1
        state[1] = 0
    pickle.dump(state, open(localpath('state.p'), 'wb'))
    
    if state[1] == 0:
        supervisor.worldLoad(localpath(worlds[state[0]]))
    else:
        supervisor.worldReload()


if __name__ == "__main__":
    main()

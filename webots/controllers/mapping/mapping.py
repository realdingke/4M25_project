"""mapping controller."""

from controller import Robot, Camera, CameraRecognitionObject, InertialUnit, DistanceSensor, PositionSensor
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

#enable distance sensors
fds = robot.getDevice('front_ds')
lds = robot.getDevice('left_ds')
rds = robot.getDevice('right_ds')
fds.enable(timestep)
lds.enable(timestep)
rds.enable(timestep)

# getting the position sensors
lps = robot.getDevice('left wheel sensor')
rps = robot.getDevice('right wheel sensor')
lps.enable(timestep)
rps.enable(timestep)

# enable camera and recognition
camera = robot.getDevice('camera1')
camera.enable(timestep)
camera.recognitionEnable(timestep)

#enable imu
imu = robot.getDevice('inertial unit')
imu.enable(timestep)

# get handler to motors and set target position to infinity
# leftMotor = robot.getDevice('left wheel motor')
# rightMotor = robot.getDevice('right wheel motor')
# leftMotor.setPosition(float('inf'))
# rightMotor.setPosition(float('inf'))
# leftMotor.setVelocity(0)
# rightMotor.setVelocity(0)

# values for robot
wheel_radius = 1.6 / 2.0
wheel_circ = 2 * 3.14 * wheel_radius
enc_unit = wheel_circ / 6.28
max_speed = 4
# distance between wheels in inches and middle of the robot d_mid
d = 2.28
d_mid = d / 2.0

# # STARTING VARIABLES FOR CONFIGURATION
# # state of robot
# dir = "North"
# # new position to keep track of x,y as the robot keeps moving
# new_pos = [15.0, -15.0]


# # converts meters to inches
# def m_to_i(meters):
#     return meters * 39.3701
#
#
# def pos_s_to_inches(val):
#     return math.fabs(val * wheel_radius)

def print_measurements():
    print(f"left: {pos_s_to_inches(lps.getValue())}, right: {pos_s_to_inches(rps.getValue())}, imu: {(imu.getRollPitchYaw()[2] * 180) / 3.14159}")


def get_p_sensors_vals():
    # returns left and right position sensors
    return pos_s_to_inches(lps.getValue()), pos_s_to_inches(rps.getValue())


def get_d_sensors_vals():
    # returns left, front, right sensors in inches
    return m_to_i(lds.getValue()), m_to_i(fds.getValue()), m_to_i(rds.getValue())


def get_time(distance, speed):
    return distance / speed


def stop_motors():
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)
    #print("Motors stopped.")


def get_rot_speed_rad(degrees, seconds, wheel_radius, d_mid):
    circle = d_mid * 2 * math.pi
    dist = (degrees / 360) * circle
    linear_vel = dist / seconds
    left_wheel_speed = linear_vel / wheel_radius
    right_wheel_speed = -1 * linear_vel / wheel_radius
    return left_wheel_speed, right_wheel_speed


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


def turn_left(ts, degrees=-90.5):
    rotate(degrees, 1.5, ts, "left")


def turn_right(ts, degrees=90.5):
    rotate(degrees, 1.5, ts, "right")


def get_robot_x_y(vals):
    global new_pos
    global last_vals
    diff = [vals[0] - last_vals[0], vals[1] - last_vals[1]]
    for i in range(len(diff)):
        diff[i] = math.fabs(diff[i])

    if math.fabs(diff[0]) >= .3 or math.fabs(diff[1]) >= .3:
            diff[0] = 0.3
            diff[1] = 0.3

    # diff average of the left and right wheel
    diff_avg = (diff[0]+ diff[1]) / 2.0

    # x and y are dependent on the direction the robot is moving in
    if dir == "North":
        x = new_pos[0]
        y = new_pos[1] + diff_avg
    elif dir == "West":
        x = new_pos[0] - diff_avg
        y = new_pos[1]
    elif dir == "East":
        x = new_pos[0] + diff_avg
        y = new_pos[1]
    elif dir == "South":
        x = new_pos[0]
        y = new_pos[1] - diff_avg
    # update the last vals
    last_vals = vals
    # store the new x and y into the new_pos
    new_pos = x, y
    return x, y


def get_direction(imu_val):
    if (imu_val <= -135 and imu_val >= -180) or (135 <= imu_val <= 180):
        dir = "North"
    elif imu_val <= -45 and imu_val > -135:
        dir = "West"
    elif 45 <= imu_val <= 135:
        dir = "East"
    elif (-45 < imu_val <= 0) or (0 <= imu_val < 45):
        dir = "South"
    return dir


def get_available_turns(walls):
    # a 0 for a wall means there is no wall in that direction
    # available_turns also returned in order [left, front, right]
    # 0, means not available, 1 means available
    # count is the number of available turns
    count = 0
    available = [0, 0, 0]
    for i in range(len(walls)):
        if walls[i] == 0:
            available[i] = 1
            count += 1
        # otherwise if a wall is there, walls[i] == 1, leave it as a 0
    return available, count


def face_north(ts):
    global dir
    while robot.step(ts) != -1:
        if dir == "West":
            # turn right to face north again
            turn_right(ts)
        elif dir == "South":
            # turn left twice to face north again
            turn_left(ts)
            turn_left(ts)
        elif dir == "East":
            # turn left to face north again
            turn_left(ts)
        if dir == "North":
            break
    dir = "North"

def compute_left_speed(l_dist, r_dist):
    if l_dist < r_dist:
        return max_speed*0.25
    else:
        return max_speed*0.5

def compute_right_speed(l_dist, r_dist):
    if l_dist < r_dist:
        return max_speed*0.5
    else:
        return max_speed*0.25


def main():
    TIME_STEP = 32

    # robot = Robot()

    # left_sensor = robot.getDevice("left_sensor")
    # right_sensor = robot.getDevice("right_sensor")
    # left_sensor.enable(TIME_STEP)
    # right_sensor.enable(TIME_STEP)
    lds = robot.getDevice('left_ds')
    rds = robot.getDevice('right_ds')
    fds.enable(timestep)
    lds.enable(timestep)

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    StepCounter = 0
    while robot.step(TIME_STEP) != -1:

        # read sensors
        left_dist = lds.getValue()
        right_dist = rds.getValue()

        # compute behavior (user functions)
        left = compute_left_speed(left_dist, right_dist)
        right = compute_right_speed(left_dist, right_dist)

        # actuate wheel motors
        left_motor.setVelocity(left)
        right_motor.setVelocity(right)

        #read camera images and display it
        img = camera.getImageArray()
        # img_l = cv.imread(img_l, cv.IMREAD_GRAYSCALE)
        # img_r = cv.imread(img_r, cv.IMREAD_GRAYSCALE)
        # # img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        # # img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        # stereo = np.concatenate((img_l, img_r), axis=1)
        # cv.imshow("forward camera image", np.array(img))

        if StepCounter%8 == 0:
            plt.imshow(img, interpolation='nearest')
            plt.show()

        StepCounter += 1

if __name__ == "__main__":
    main()
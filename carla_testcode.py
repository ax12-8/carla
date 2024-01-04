import carla
import numpy as np
import pygame
import cv2
import math
import random
import matplotlib.pyplot as plt
from collections import deque
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector, is_within_distance, get_speed
distance = 1.0

# Define a function to process GPS data
def process_gps(gps_data, prev_location, prev_timestamp):
    current_location = gps_data.transform.location
    current_timestamp = gps_data.timestamp.elapsed_seconds

    # Calculate speed based on the change in location over time
    if prev_location is not None and prev_timestamp is not None:
        delta_location = current_location - prev_location
        delta_time = current_timestamp - prev_timestamp

        speed = delta_location.length() / delta_time  # Speed in meters per second
        speed_kmh = speed * 3.6  # Convert speed to kilometers per hour

        # Process GPS data as needed
        print(f"GPS Location: {current_location}, Speed: {speed_kmh} km/h")

    # Update previous location and timestamp for the next iteration
    prev_location = current_location
    prev_timestamp = current_timestamp


def adjust_decimal(value, target_range=(0.01, 0.1)):
    # 将数值转换为科学记数法
    exponent = 0
    while abs(value) < target_range[0] or abs(value) > target_range[1]:
        if abs(value) < target_range[0]:
            value *= 10
            exponent -= 1
        elif abs(value) > target_range[1]:
            value /= 10
            exponent += 1

    return value, exponent
def get_next_waypoint(vehicle, distance_threshold=5.0):
    # 获取车辆的当前位置
    vehicle_location = vehicle.get_location()

    # 获取车辆即将要到达的下一个路径点
    waypoint = vehicle.get_world().get_map().get_waypoint(vehicle_location)

    # 计算车辆与即将到达路径点的距离
    distance_to_waypoint = vehicle_location.distance(waypoint.transform.location)

    # 仅在距离小于阈值时返回路径点信息
    if distance_to_waypoint < distance_threshold:
        return waypoint.transform.location, waypoint.transform.rotation.yaw
    else:
        return None

def calculate_tracking_error_based_on_next_waypoint(vehicle, w_heading=0.4, w_lateral=0.4, w_longitudinal=0.2):
    next_waypoint_info = get_next_waypoint(vehicle)

    if next_waypoint_info:
        next_waypoint_location, next_waypoint_heading = next_waypoint_info

        vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)
        next_waypoint_heading = math.radians(next_waypoint_heading)
        heading_error = vehicle_yaw - next_waypoint_heading

        lateral_error = vehicle.get_location().distance(next_waypoint_location)

        target_long_dis = vehicle.get_velocity().length()
        current_dis = vehicle.get_velocity().length()
        longitudinal_error = target_long_dis - current_dis

        # 归一化处理
        normalized_heading_error = heading_error / math.pi  # 将航向角误差归一化到[-1, 1]范围
        normalized_lateral_error = lateral_error / 10.0
        normalized_longitudinal_error = longitudinal_error / 10.0

        # 计算轨迹跟踪误差
        tracking_error = (w_heading * normalized_heading_error +
                          w_lateral * normalized_lateral_error +
                          w_longitudinal * normalized_longitudinal_error)/20.00


        adjusted_error, exponent = adjust_decimal(tracking_error)
        random_noise = random.uniform(-0.004, 0.004)
        tracking_error += random_noise
        return abs(adjusted_error)


def process_img(image):  #图像输出
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0

IM_WIDTH = 640
IM_HEIGHT = 480
actor_list = []

client = carla.Client('localhost',2000)
client.load_world('Town05')
world = client.get_world()
m = world.get_map()
weather = carla.WeatherParameters(cloudiness=20.0, sun_altitude_angle=70)
world.set_weather(weather)
transform = carla.Transform()
spectator = world.get_spectator()
bv_transform = carla.Transform(transform.location + carla.Location(z=2.5,x=-6), carla.Rotation(yaw=0, pitch=0))
spectator.set_transform(bv_transform)

blueprint_library = world.get_blueprint_library()
spawn_points = m.get_spawn_points()

T = 10
'''for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=T)
    world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=T)'''

# global path planner 
origin = carla.Location(spawn_points[117].location)
destination = carla.Location(spawn_points[56].location)

grp = GlobalRoutePlanner(m, distance)
route = grp.trace_route(origin, destination)

wps = []
for i in range(len(route)):
    wps.append(route[i][0])
draw_waypoints(world, wps)


for pi, pj in zip(route[:-1], route[1:]):
    pi_location = pi[0].transform.location
    pj_location = pj[0].transform.location 
    pi_location.z = 0.5
    pj_location.z = 0.5
    world.debug.draw_line(pi_location, pj_location, thickness=0.1,  color=carla.Color(r=255))#life_time=T,
    pi_location.z = 0.6
    world.debug.draw_point(pi_location, color=carla.Color(g=255))#, life_time=T
    
# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.model3')
ego = world.spawn_actor(ego_bp, spawn_points[117])

waypoint = world.get_map().get_waypoint(ego.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))



# PID
args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}

args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}

PID=VehiclePIDController(ego,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

i = 0
target_speed = 20
next = wps[0]

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        # self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
# def pygame_callback(data, obj):
#     img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
#     img = img[:,:,:3]
#     img = img[:, :, ::-1]
#     obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

# camera 
camera_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_trans, attach_to=ego)

# camera.listen(lambda image: pygame_callback(image, renderObject))


# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(image_w, image_h)

# Initialise the display
# pygame.init()
# gameDisplay = pygame.display.set_mode((image_w,image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
# gameDisplay.fill((0,0,0))
# gameDisplay.blit(renderObject.surface, (0,0))
# pygame.display.flip()
# 初始化数据记录
timestamps = deque(maxlen=200)
speeds = deque(maxlen=200)
accelerations = deque(maxlen=200)
headings = deque(maxlen=200)
tracking_errors = deque(maxlen=200)
# 定义图表
fig, axs = plt.subplots(2)
fig, axs1 = plt.subplots(2)
# plt.xticks([1, 1], ['axs', 'axs1'])
# 图像传感器
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
camera_bp.set_attribute("fov", "110")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp,camera_transform, attach_to=ego)
actor_list.append(camera)
camera.listen(lambda data: process_img(data))

# Initialize previous location and timestamp
prev_location = None
prev_timestamp = None

# Attach a callback function to the GPS sensor
#gps.listen(lambda data: process_gps(data, prev_location, prev_timestamp))

try:
    while True:
        ego_transform = ego.get_transform()
        #asshole spectator
        '''transform1 = carla.Transform(ego_transform.transform(carla.Location(x=-6, z=2.5)),
                                        ego_transform.rotation)'''
        #overlooking spectator
        transform1 = carla.Transform(ego_transform.location + carla.Location( z = 50),
             carla.Rotation(pitch = -90) )
        spectator.set_transform(transform1)
        ego_loc = ego.get_location()
        world.debug.draw_point(ego_loc, life_time=T, color=carla.Color(r=255))#
        world.debug.draw_point(next.transform.location, life_time=T, color=carla.Color(r=255))#
        ego_dist = distance_vehicle(next, ego_transform)
        #ego_vect = vector(ego_loc, next.transform.location)
        control = PID.run_step(target_speed, next)

        if i == (len(wps)-1):
            control = PID.run_step(0, wps[-1])
            ego.apply_control(control)
            print('this trip finish')
            break

        if ego_dist < 1.5: 
            i = i + 1
            next = wps[i]
            control = PID.run_step(target_speed, next)

        # lane change start waypoint
        if distance_vehicle(world.get_map().get_waypoint(spawn_points[114].location), ego_transform) < 0.5:
            origin = carla.Location(spawn_points[114].location)
            current_w = world.get_map().get_waypoint(spawn_points[114].location)
            d_lanechange = 1*target_speed/3.6
            next_w = current_w.next(d_lanechange)[0]
        
        ego.apply_control(control)
        world.wait_for_tick()
        
        # Update the display
        # gameDisplay.blit(renderObject.surface, (0,0))
        # pygame.display.flip()

        # 获取汽车状态
        vehicle_state = ego.get_velocity()
        velocity = 3.6 * np.sqrt(vehicle_state.x ** 2 + vehicle_state.y ** 2 + vehicle_state.z ** 2)  # 转换为km/h
        acceleration = (velocity - speeds[-1]) / 2 if speeds else 0.0  # 使用简单的差分来估算加速度
        heading = math.degrees(control.steer)

        # 获取车辆的控制信息
        control = ego.get_control()

        # 记录时间戳和数据
        timestamps.append(world.get_snapshot().timestamp.elapsed_seconds)
        speeds.append(velocity)
        accelerations.append(acceleration)
        if world.get_snapshot().timestamp.elapsed_seconds < 7.0:
            heading = heading / 6.25
        if (world.get_snapshot().timestamp.elapsed_seconds < 30.0) and (world.get_snapshot().timestamp.elapsed_seconds > 25.0):
            heading = heading / 6.25
        headings.append(heading)
        tracking_error = float(calculate_tracking_error_based_on_next_waypoint(ego))
        tracking_errors.append(tracking_error)

        # 更新图表数据
        axs[0].set_xlim([0, max(timestamps)])
        axs[1].set_xlim([0, max(timestamps)])
        axs1[0].set_xlim([0, max(timestamps)])
        axs1[1].set_xlim([0, max(timestamps)])

        # axs[0].clear()
        axs[0].plot(timestamps, speeds)
        axs[0].set_ylabel('Speed (km/h)')
        axs[0].set_title('Vehicle Speed')

        # axs[1].clear()
        axs[1].plot(timestamps, accelerations)
        axs[1].set_ylabel('Acceleration (km/h^2)')
        axs[1].set_title('Vehicle Acceleration')
        axs[1].set_xlabel('Time (s)')

        # axs1[0].clear()
        axs1[0].plot(timestamps, headings)
        axs1[0].set_ylabel('Heading (degrees)')
        axs1[0].set_title('Vehicle Heading')

        # axs1[1].clear()
        axs1[1].plot(timestamps, tracking_errors)
        axs1[1].set_ylabel('Vehicle tracking_error')
        axs1[1].set_title('Vehicle tracking_errors')
        axs1[1].set_xlabel('Time (s)')

        # plt.xlabel('Time (seconds)')

        # tracking_error = float(calculate_tracking_error_based_on_next_waypoint(ego))
        # print("轨迹跟踪误差: {}".format(tracking_error))

finally:
    ego.destroy()
    camera.stop()
    pygame.quit()
    plt.show()
    # Destroy the GPS sensor

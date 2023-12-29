import carla
import math
import cv2
import pygame
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import os

def process_img(image):  # 图像输出
    i = np.array(image.raw_data)
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3 / 255.0


IM_WIDTH = 640
IM_HEIGHT = 480
actor_list = []
# counter = 0
# max_images = 50
# output_directory = "C:/Users/abcd/Desktop/map1"  # 修改为实际的输出目录


from agents.navigation.global_route_planner import GlobalRoutePlanner

# def lidar_callback(data):
#     global counter
#     global output_directory
#
#     # 检查是否达到存储数量上限
#     if counter >= 50:
#         print("Reached maximum storage limit. Stopping further storage.")
#         return
#
#     lidar_data = data
#     points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
#
#     # 确保点云数据的数量是3的倍数
#     num_points = len(points) // 3
#     points = np.reshape(points[:num_points * 3], (num_points, 3))
#
#     # 3D点云图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='.')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'Point Cloud {counter}')
#
#     # 保存3D点云图
#     img_path = os.path.join(output_directory, f"point_cloud_3d_{counter}.png")
#     plt.savefig(img_path)
#     plt.show()
#
#     counter += 1
#
#     # 检查是否达到存储数量上限
#     if counter >= 50:
#         print("Reached maximum storage limit. Stopping further storage.")
#         return

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

def calculate_tracking_error_based_on_next_waypoint(vehicle,w_heading=0.4, w_lateral=0.6):
    # 获取车辆即将要到达的下一个路径点
    next_waypoint_info = get_next_waypoint(vehicle)

    # 仅当路径点信息可用时才进行计算
    if next_waypoint_info:
        next_waypoint_location, next_waypoint_heading = next_waypoint_info

        # 计算航向角误差
        vehicle_yaw = math.radians(vehicle.get_transform().rotation.yaw)
        next_waypoint_heading = math.radians(next_waypoint_heading)
        heading_error = vehicle_yaw - next_waypoint_heading

        # 计算横向误差
        lateral_error = vehicle.get_location().distance(next_waypoint_location)

        # 计算轨迹跟踪误差
        tracking_error = (w_heading*heading_error + w_lateral*lateral_error)/100
        random_noise = random.uniform(-0.03, 0.03)
        tracking_error += random_noise

        return abs(tracking_error)

def main():
    try:
        # 连接到Carla服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # 改变地图Town01
        client.load_world('Town01')
        # 获取世界
        world = client.get_world()
        # 改变天气
        weather = carla.WeatherParameters(cloudiness=20.0, sun_altitude_angle=70)
        world.set_weather(weather)
        # 获取地图
        current_map = world.get_map()
        spawn_points = world.get_map().get_spawn_points()
        # 获取道路信息
        roads = current_map.get_topology()
        for way_point in roads[5]:
            print(way_point.transform)
        # 获取地图的waypoint
        waypoints = current_map.generate_waypoints(distance=2.0)
        sampling_resolution = 2
        grp = GlobalRoutePlanner(current_map, sampling_resolution)
        pnt_start = carla.Location(x=384.591064, y=1.980000, z=0.600000)
        pnt_end = carla.Location(x=348.231079, y=1.999316, z=0.600000)

        route = grp.trace_route(pnt_start, pnt_end)

        type(route)

        for pnt in route:
            world.debug.draw_string(pnt[0].transform.location, '`', draw_shadow=False, persistent_lines=True)
            world.debug.draw_point(pnt[0].transform.location, color=carla.Color(0, 255, 0, 0))

        # 打印waypoint的位置
        '''for waypoint in route:
            print("Waypoint: {}, Location: {}".format(waypoint.id, waypoint.transform.location))'''

        for spawn_point in spawn_points:
            world.debug.draw_string(spawn_point.location, 'o', draw_shadow=False, persistent_lines=True)
            world.debug.draw_point(spawn_point.location)
            if (spawn_point == spawn_points[2]):
                world.debug.draw_point(spawn_point.location, color=carla.Color(0, 255, 255, 0))
        # 获取蓝图
        blueprint_library = world.get_blueprint_library()
        model3_bp = blueprint_library.filter("model3")[0]
        # 生成model3
        vehicle = world.try_spawn_actor(model3_bp, spawn_points[2])
        vehicle.set_autopilot(True)
        # vehicle.set_transform()
        # 初始化数据记录
        timestamps = deque(maxlen=200)
        speeds = deque(maxlen=200)
        accelerations = deque(maxlen=200)
        headings = deque(maxlen=200)
        # 定义图表
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        # 定义图表
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        # 图像传感器
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda data: process_img(data))
        # GPS
        gps_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gps_transform = carla.Transform(carla.Location(x=1.0, y=0.0, z=2.8))
        gps_sensor = world.spawn_actor(gps_bp, gps_transform, attach_to=vehicle)
        # 添加激光雷达传感器
        # lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        # lidar_transform = carla.Transform(carla.Location(x=0.5, z=2.4))
        # lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        # actor_list.append(lidar_sensor)
        # lidar_sensor.listen(lambda data: lidar_callback(data))
        while True:
            # set the sectator to follow the ego vehicle
            spectator = world.get_spectator()
            # transform = vehicle.get_transform()
            transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-6, z=2.5)),
                                        vehicle.get_transform().rotation)
            spectator.set_transform(transform)
            # spectator.set_transform(carla.Transform(transform.location + carla.Location( z = 20),
            # carla.Rotation(pitch = transform.rotation.pitch-90,
            # yaw = transform.rotation.yaw)))
            #GPS传感器数据输出
            '''gps_data = gps_sensor.get_transform()
            print("GPS Coordinates: ({}, {}, {})".format(gps_data.location.x, gps_data.location.y, gps_data.location.z))'''
            # 获取汽车状态
            vehicle_state = vehicle.get_velocity()
            velocity = (3.6 * np.sqrt(
                vehicle_state.x ** 2 + vehicle_state.y ** 2 + vehicle_state.z ** 2)) / 2  # 转换为km/h
            acceleration = (velocity - speeds[-1]) / 0.1 if speeds else 0.0  # 使用简单的差分来估算加速度
            heading = vehicle.get_transform().rotation.yaw
            # 记录时间戳和数据
            timestamps.append(world.get_snapshot().timestamp.elapsed_seconds)
            speeds.append(velocity)
            accelerations.append(acceleration)
            headings.append(heading)
            #输出速度加速度
            print("model3 velocity:({})".format(velocity))
            print("model3 acceleration:({})".format(acceleration))
            # 计算轨迹跟踪误差
            tracking_error = calculate_tracking_error_based_on_next_waypoint(vehicle)

            print("轨迹跟踪误差: {:.2f}".format(tracking_error))

            # 更新图表数据
            '''axs[0].clear()
            axs[0].plot(timestamps, speeds)
            axs[0].set_ylabel('Speed (km/h)')
            axs[0].set_title('Vehicle Speed')

            axs[1].clear()
            axs[1].plot(timestamps, accelerations)
            axs[1].set_ylabel('Acceleration (km/h^2)')
            axs[1].set_title('Vehicle Acceleration')

            axs[2].clear()
            axs[2].plot(timestamps, headings)
            axs[2].set_ylabel('Heading (degrees)')
            axs[2].set_title('Vehicle Heading')

            plt.xlabel('Time (seconds)')

            plt.pause(0.1)'''


    finally:
        for actor in world.get_actors().filter('*vehicle*'):
            actor.destroy()
        pass


if __name__ == '__main__':
    main()

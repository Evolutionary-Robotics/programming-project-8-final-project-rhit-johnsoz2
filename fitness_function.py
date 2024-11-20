import math

import numpy as np
import pybullet as p
import time


def fitness_calc(agent, initial_position, speeds, task1_weight, task2_weight, task3_weight):
    # total_weight = task1_weight + task2_weight + task3_weight
    # task1_weight /= total_weight
    # task2_weight /= total_weight
    # task3_weight /= total_weight

    # define task-specific fitness function
    # 1: how far the body moves in a straight line (angle offset stuff)
    task1_score = straight_movement_score(agent, initial_position) * task1_weight
    # 2: stability of body (tilting, instability bad, based on average height of body's center of mass, penalize if body tips over/rotates too much)
    task2_score = body_stability(agent) * task2_weight
    # 3: consistency of speed over time (average speed over time, reward consistency)
    task3_score = speed_over_time(speeds) * task3_weight

    # print()
    # print(f"Task 1 Score: {task1_score}")
    # print(f"Task 2 Score: {task2_score}")
    # print(f"Task 3 Score: {task3_score}")

    return task1_score + task2_score + task3_score


def straight_movement_score(agent, initial_position):
    final_position = p.getBasePositionAndOrientation(agent)[0]
    # distance = np.linalg.norm(np.array(final_position) - np.array(initial_position))
    # return distance
    distance_vector = np.array(final_position) - np.array(initial_position)
    straightness = abs(distance_vector[0]) / np.linalg.norm(distance_vector)  # Assuming movement in x-direction
    return np.linalg.norm(distance_vector) * straightness


def body_stability(agent):
    # link_index = 1
    # state = p.getLinkState(agent, link_index)
    # orientation = state[1]
    base_orientation = p.getBasePositionAndOrientation(agent)[1]
    euler = p.getEulerFromQuaternion(base_orientation)
    roll = np.degrees(abs(euler[0]))
    pitch = np.degrees(abs(euler[1]))
    tilt = np.sqrt((roll * roll) + (pitch * pitch))
    tilt_penalty = 90   # was 90
    return max(0, 1 - (tilt / tilt_penalty))


def speed_over_time(speeds):
    speed_variance = np.std(speeds)
    score = 1 / speed_variance
    # score = 1 - np.tanh(speed_variance)
    if score == math.inf:
        return 0
    return score

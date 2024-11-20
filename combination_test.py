import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as ps
import time
import matplotlib.pyplot as plt
import elephant_ctrnn
from r2d2controller import Controller
import r2d2body
import r2d2ctrnn
import fitness_function as ef


def elephant_movement_sim_CTRNN(num_legs, step_height, robot_id, nn_outputs):
    target_positions = []

    # Use NN outputs to control leg positions
    for j in range(num_legs):
        # Scale the output if needed, or directly use it
        target_positions.append(nn_outputs[j] * step_height)

    # Set joint positions for walking motion
    p.setJointMotorControlArray(robot_id,
                                range(num_legs),
                                p.POSITION_CONTROL,
                                targetPositions=target_positions,
                                forces=[500] * num_legs)


def initialize_physics_engine(do_GUI):
    if do_GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, rgbBackground=[0, 0, 50])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    camera_distance = 12  # Adjust this value to zoom in/out
    camera_pitch = -30  # Adjust the pitch (vertical angle)
    camera_yaw = 40  # Adjust the yaw (horizontal angle)
    camera_target = [0, 0, 1]  # The point the camera looks at (x, y, z)

    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)


def mutate_nn(nn):
    mutated_nn = nn.clone()
    random_weight_change = np.random.randn(*nn.weights.shape) * 0.25
    mutated_nn.weights += random_weight_change
    return mutated_nn


def one_sim(duration, ectrnn, rctrnn, e_compare, iteration):
    if iteration == 0 or iteration == TRIALS:
        initialize_physics_engine(True)
    else:
        initialize_physics_engine(False)

    p.loadURDF("plane.urdf")
    elephant_id = p.loadURDF("elephant.urdf")
    num_legs = 4
    step_height = 0.5
    ps.Prepare_To_Simulate(elephant_id)
    p.setJointMotorControlArray(elephant_id,
                                range(num_legs),
                                p.POSITION_CONTROL,
                                targetPositions=[0] * num_legs,
                                forces=[500] * num_legs)

    ebfit = 0
    r2d2_bfit = 0

    enn = 0
    if ectrnn == e_compare:
        enn = elephant_ctrnn.CTRNN(num_legs)
        enn.load(ectrnn)
        enn.initialize_state(np.zeros(num_legs))
    else:
        enn = ectrnn
    espeeds = []

    r2d2num_neurons = 15
    r2d2_body = r2d2body.Body("r2d2.urdf")

    if rctrnn == 'r2d2.urdf':
        r2d2_ctrnn = r2d2ctrnn.CTRNN(r2d2num_neurons)
    else:
        r2d2_ctrnn = rctrnn
    r2d2controller = Controller(r2d2_body, r2d2_ctrnn)
    joint_trajectories = []
    r2d2_speeds = []

    for i in range(duration):
        foot_sensors = [
            ps.Get_Touch_Sensor_Value_For_Link(f'1'),
            ps.Get_Touch_Sensor_Value_For_Link(f'2'),
            ps.Get_Touch_Sensor_Value_For_Link(f'3'),
            ps.Get_Touch_Sensor_Value_For_Link(f'4')
        ]

        elephant_initial_position = p.getBasePositionAndOrientation(elephant_id)[0]
        r2d2_initial_position = p.getBasePositionAndOrientation(r2d2_body.body_id)[0]

        # Prepare inputs for the CTRNN
        enn.inputs = np.array(foot_sensors)  # Sensor values as inputs to the network

        # Step the CTRNN and get outputs
        dt = 0.01  # Define your time step
        enn.step(dt, ebfit)  # Update the CTRNN
        enn_outputs = enn.outputs  # Get the outputs from the CTRNN

        joint_positions = r2d2controller.body.get_joint_positions()

        # Use the joint positions as inputs to the CTRNN
        ctrnn_output = r2d2_ctrnn.step(joint_positions, r2d2_bfit)

        # Apply control signals from the CTRNN output
        r2d2controller.body.apply_joint_control(ctrnn_output)

        joint_positions = r2d2controller.body.get_joint_positions()
        joint_trajectories.append(joint_positions)

        # Update robot movement based on NN output
        elephant_movement_sim_CTRNN(num_legs, step_height, elephant_id, enn_outputs)

        p.stepSimulation()
        r2d2_body.step_simulation()

        # tracking speed
        espeed = np.linalg.norm(np.array(p.getBasePositionAndOrientation(elephant_id)[0]) - np.array(
            elephant_initial_position)) / dt
        espeeds.append(espeed)

        r2d2_speed = np.linalg.norm(np.array(p.getBasePositionAndOrientation(r2d2_body.body_id)[0]) - np.array(
            r2d2_initial_position)) / dt
        r2d2_speeds.append(r2d2_speed)

        efitness = ef.fitness_calc(elephant_id, elephant_initial_position, espeeds,
                                   ELEPHANT_FITNESS_WEIGHTS[0], ELEPHANT_FITNESS_WEIGHTS[1], ELEPHANT_FITNESS_WEIGHTS[2])
        r2d2_fitness = ef.fitness_calc(r2d2_body.body_id, r2d2_initial_position, r2d2_speeds,
                                       R2D2_FITNESS_WEIGHTS[0], R2D2_FITNESS_WEIGHTS[1], R2D2_FITNESS_WEIGHTS[2])

        if ebfit < efitness:
            ebfit = efitness

        if r2d2_bfit < r2d2_fitness:
            r2d2_bfit = r2d2_fitness

        time.sleep(1 / 500)

    p.disconnect()
    return ebfit, r2d2_bfit, enn, r2d2_ctrnn


def hill_climbing_simulation(elephant_ctrnn_file, r2d2_ctrnn_file):
    elephant_fitnesses = []
    r2d2_fitnesses = []

    elephant_parent_fitness, r2d2_parent_fitness, elephant_parent_nn, r2d2_parent_nn \
        = one_sim(DURATION, elephant_ctrnn_file, r2d2_ctrnn_file, elephant_ctrnn_file, 0)

    elephant_fitnesses.append(elephant_parent_fitness)
    r2d2_fitnesses.append(r2d2_parent_fitness)

    for i in range(TRIALS):
        mutated_elephant_nn = mutate_nn(elephant_parent_nn)
        mutated_r2d2_nn = mutate_nn(r2d2_parent_nn)

        elephant_child_fitness, r2d2_child_fitness, elephant_child_nn, r2d2_child_nn \
            = one_sim(DURATION, elephant_ctrnn_file, r2d2_ctrnn_file, elephant_ctrnn_file, i + 1)

        if elephant_parent_fitness < elephant_child_fitness:
            elephant_parent_nn = mutated_elephant_nn
            elephant_parent_fitness = elephant_child_fitness
        if r2d2_parent_fitness < r2d2_child_fitness:
            r2d2_parent_nn = mutated_r2d2_nn
            r2d2_parent_fitness = r2d2_child_fitness
        elephant_fitnesses.append(elephant_child_fitness)
        r2d2_fitnesses.append(r2d2_child_fitness)
        print(f"Trial {i + 1} finished!\n")

    best_elephant_fitness = elephant_parent_fitness
    best_r2d2_fitness = r2d2_parent_fitness

    return best_elephant_fitness, best_r2d2_fitness, elephant_fitnesses, r2d2_fitnesses


def plot_hill_climbing_results(e_fot, r_fot):
    iterations = list(range(1, len(e_fot) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, e_fot, label="Elephant", marker='o', linestyle='-')
    plt.plot(iterations, r_fot, label="R2D2", marker='x', linestyle='--')
    plt.title("Fitness comparison of Elephant and R2D2 Bodies over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_elephant_hill_climbing_results(e_fot):
    iterations = list(range(1, len(e_fot) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, e_fot, label="Elephant", marker='o', linestyle='-')
    plt.title("Fitness comparison of Elephant Bodies over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


# Global Vars
DURATION = 2000
TRIALS = 3
ELEPHANT_FITNESS_WEIGHTS = [0.3, 0.6, 0.1]
R2D2_FITNESS_WEIGHTS = [0.05, 0.05, 0.9]

bef, brf, aef, arf = hill_climbing_simulation('ctrnn_3.npz', 'r2d2.urdf')
print(f"Best Elephant Fitness: {bef}")
print(f"Best R2D2 Fitness: {brf}")
plot_hill_climbing_results(aef, arf)
plot_elephant_hill_climbing_results(aef)

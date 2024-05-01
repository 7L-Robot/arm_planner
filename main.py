import argparse
import numpy as np

from planner import Planner

from simulator import RobotEnv, Camera, add_box

import pybullet
import pybullet_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--plan', '-p', type=str, default='rrtc', help="Use [  rrt | rrtc | prm | obprm ]")
    parser.add_argument('--map', '-m', type=int, default=2, help="Use map [ 1 | 2 | 3 ]")
    parser.add_argument('--reuse_graph', '-reuse_graph', type=str2bool, const=True, nargs='?', default=False, help="Reuse the graph for PRM?")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    urdf_path = "./data/franka_description/robots/panda_arm_hand.urdf"
    srdf_path = "./data/franka_description/robots/panda_arm_hand.srdf"

    joint_groups = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    ee_joint = 'panda_ee_joint'


    urdf_path = "./data/UR5/urdf/ur5e.urdf"
    srdf_path = "./data/UR5/urdf/ur5e.srdf"
    joint_groups = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ee_joint = 'wrist_3_joint'
    
    # ee_joint = 'ee_fixed_joint'


    physcisClient = pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeID = pybullet.loadURDF("plane.urdf")
    bid = pybullet.loadURDF(urdf_path, useFixedBase=True, flags=pybullet.URDF_USE_SELF_COLLISION)
    # 加相机
    cam = Camera(640, 480, 57, 3, [1.5, -0.1, 0.4], [0,0,0.37], [-1, 0, 0], 0.055, 10)

    # 加机器人
    bot = RobotEnv(bid, cam, True, True)


    planner = Planner(urdf_path, srdf_path, joint_groups, ee_joint, args.plan)

    '''
    TODO: Replace obstacle box w/ the box specs in your workspace:
    [x, y, z, r, p, y, sx, sy, sz]
    '''
    if args.map == 3:
        boxes = np.array([
            # obstacle
            # [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.45, -0.45, 0.7, 0, 0, 0.78, 0.6, 0.6, 0.05],
            # sides
            [-0.7, 0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
            [0.7, -0.7, 0.75, 0, 0, 0.78, 2, 0.01, 1.6],
            # back
            [-0.7, -0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
            # front
            # [0.7, 0.7, 0.75, 0, 0, 0.78, 0.01, 2, 1.6],
            # top
            [0, 0, 1.5, 0, 0, 0.78, 2, 2, 0.01],
            # bottom
            [0, 0, -0.08, 0, 0, 0.78, 2, 2, 0.01]
        ])
    elif args.map == 2:
        boxes = np.array([
            # obstacle
            [0.7, 0, 0.6, 0, 0, 0, 0.45, 0.3, 0.05],
            # sides
            [0.15, 0.66, 0.65, 0, 0, 0, 1.2, 0.01, 1.5],
            [0.15, -0.66, 0.65, 0, 0, 0, 1.2, 0.01, 1.5],
            # back
            [-0.41, 0, 0.65, 0, 0, 0, 0.01, 1.4, 1.5],
            # front
            # [0.75, 0, 0.65, 0, 0, 0, 0.01, 1.4, 1.5],
            # top
            [0.2, 0, 1.35, 0, 0, 0, 1.2, 1.4, 0.01],
            # bottom
            [0.2, 0, -0.08, 0, 0, 0, 1.2, 1.4, 0.01]
        ])
    else:
        boxes = np.array([
            # obstacle
            # [0, 0, 0, 0, 0, 0, 0, 0, 0],
            # [0.4, 0, 0.25, 0, 0, 0, 0.3, 0.05, 0.5],
            
            # sides
            [0.15, 0.6, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
            [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
            # back
            [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
            # front
            # [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],

            # top
            [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
            # bottom
            [0.2, 0, -0.08, 0, 0, 0, 1.2, 1, 0.01]
        ])



    '''
    TODO: Fill in start and target joint positions
    '''
    if args.map == 3:
        joints_start = np.array([0, 3*np.pi/8, 0, -np.pi / 8, 0, np.pi / 2, np.pi / 4])
        joints_start[0] = -np.deg2rad(45)
        joints_target = np.array([0, 0, 0, -np.pi / 4, 0, np.pi / 4, np.pi / 4])
        
        joints_target = np.array([0, -np.pi/2, np.pi/4, -np.pi/4, -np.pi/2, 0])
        
        joints_target[0] = -np.deg2rad(45)
    elif args.map == 2:
        joints_start = np.array([0, np.pi/6, 0, -2*np.pi / 3, 0, 5*np.pi / 6, np.pi / 4])
        joints_target = np.array([0, 0, 0, -np.pi / 4, 0, np.pi / 4, np.pi / 4])
        
        if 'ur' in urdf_path:
            joints_start = np.array([0, -np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0])
            joints_target = np.array([0, -np.pi/2, np.pi/4, -np.pi/4, -np.pi/2, 0])

    else:
        joints_start = planner.robot.home_joints.copy()
        joints_start[0] = -np.deg2rad(45)
        joints_target = joints_start.copy()
        joints_target[0] = np.deg2rad(45)
    

    for box in boxes:
        # add_box( box[-3:]/2.0, box[:3], color=[1,1,0,0.5] )
        planner.add_obstacles(box)

    bot.set_joints(joints_start)

    def ee_upright_constraint(q):
        '''
        TODO: Implement constraint function and its gradient.

        This constraint should enforce the end-effector stays upright.
        Hint: Use the roll and pitch angle in desired_ee_rp. The end-effector is upright in its home state.

        Input:
            q - a joint configuration

        Output:
            err - a non-negative scalar that is 0 when the constraint is satisfied
            grad - a vector of length 6, where the ith element is the derivative of err w.r.t. the ith element of ee
        '''
        ee = planner.robot.ee(q)
        
        # desired_ee_rp = planner.robot.ee(joints_start)[3:4]
        # err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:4])) ** 2)
        # grad = np.asarray([0, 0, 0, 2 * (ee[3] - desired_ee_rp[0]), 0, 0])
        desired_ee_rp = planner.robot.ee(joints_start)[3:5]
        err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:5])) ** 2)
        grad = np.asarray([0, 0, 0, 2 * (ee[3] - desired_ee_rp[0]), 2 * (ee[4] - desired_ee_rp[1]), 0])
        return err, grad

    constraint = ee_upright_constraint
    constraint = None

    # add_box( [0.1, 0.01, 0.02], pos, quat, color=[0,1,0,1] )

    if True:
        plan = planner.plan(joints_start, joints_target, constraint, args)
        plan = np.array(plan)
        # plan = np.concatenate([plan, plan[:,:2]], axis=1)

        bot.show_path(plan)
        input('go')
        bot.follow_path(plan, 10)

    input('end')
    a = 1

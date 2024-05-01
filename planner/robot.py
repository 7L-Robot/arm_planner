import xml.dom
import xml.dom.minidom
import numpy as np
# import quaternion
from scipy.spatial.transform import Rotation
from itertools import product
import xml

from yourdfpy import urdf

import pybullet

import time
def timer(func):
    def wrapper(*args, **kw):
        st = time.time()        
        ret = func(*args, **kw)
        et = time.time()        
        print('cost: ', et - st)
        return ret
    return wrapper

class Robot:

    def __init__(self, urdf_path, srdf_path=None, joint_groups=[], ee_joint=None ):

        self.urdf_path = urdf_path
        self.srdf_path = srdf_path

        self.joint_groups = joint_groups

        self.num_dof = len(self.joint_groups)        

        if ee_joint is None:
            self.ee_joint = self.joint_groups[-1]
        else:
            self.ee_joint = ee_joint

        self.bullet_client = pybullet.connect(pybullet.DIRECT)
        self.bullet_robot = pybullet.loadURDF(urdf_path, useFixedBase=True)
        joint_num = pybullet.getNumJoints(self.bullet_robot)

        joints = []
        names = []

        joint_limits_lower = []
        joint_limits_upper = []
        joint_velocities = []
        
        for j in range(joint_num):
            info = pybullet.getJointInfo(self.bullet_robot, j)
            joint_name = info[1].decode()
            joint_type = info[2]

            joint_low = info[8]
            joint_high = info[9]
            joint_vel = info[11]

            if joint_type in [0, 1]:
                joints.append(j)
                names.append(joint_name)

                if joint_name in joint_groups:
                    joint_velocities.append(joint_vel)
                    joint_limits_lower.append(joint_low)
                    joint_limits_upper.append(joint_high)
            
            if joint_name == ee_joint:
                self.ee_joint_id = j

        self.bullet_joints = joints
        self.bullet_joint_names = names
        
        self.joint_limits_low = np.array(joint_limits_lower)
        self.joint_limits_high = np.array(joint_limits_upper)
        self.joint_velocities = np.array(joint_velocities)

        self.bullet_obstacles = {}
        self.collision_min_dist = 0.03

        self.load_links(srdf_path)

        if self.num_dof == 7:
            self.home_joints = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
        else:    
            self.home_joints = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])


    def get_joints(self):
        all_joints = pybullet.getJointStates(self.bullet_robot, self.bullet_joints)
        joint_poses = []
        for joint in all_joints:
            joint_poses.append(joint[0])
        return np.array(joint_poses)


    def forward_kinematics(self, joints):
        '''
        Calculate the position of each joint using the dh_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        self.set_joints(joints)

        if self.ee_joint_id in self.bullet_joints:
            all_joints = self.bullet_joints
        else:
            all_joints = self.bullet_joints + [self.ee_joint_id]

        fk_mat = np.zeros( (len(all_joints), 4, 4) )
        
        for j, joint_id in enumerate(all_joints):
            
            link_state = pybullet.getLinkState(self.bullet_robot, joint_id)
            pos, quat = link_state[:2]
            mat = np.eye(4)
            mat[:3,:3] = Rotation.from_quat(quat).as_matrix()
            mat[:3,3] = pos

            fk_mat[j] = mat

        return fk_mat

    def load_links(self, srdf_path):

        # read srdf_path
        disable_dict = {}
        if srdf_path is not None:
            dom = xml.dom.minidom.parse(srdf_path)
            data = dom.getElementsByTagName('disable_collisions')
            for d in data:
                l1 = d.attributes['link1'].value
                l2 = d.attributes['link2'].value
                if l1 not in disable_dict:
                    disable_dict[l1] = []
                if l2 not in disable_dict:
                    disable_dict[l2] = []

                disable_dict[l2].append(l1)
                disable_dict[l1].append(l2)
        
        _link_name_to_index = {pybullet.getBodyInfo(self.bullet_robot)[0].decode('UTF-8'):-1,}
        _link_ids = []
        _neighbor_links = {}
        
        for _id in range(pybullet.getNumJoints(self.bullet_robot)):
            _name = pybullet.getJointInfo(self.bullet_robot, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
            _link_ids.append(_id)

        for _id in range(pybullet.getNumJoints(self.bullet_robot)):
            joint_info = pybullet.getJointInfo(self.bullet_robot, _id)
            _name = joint_info[12].decode('UTF-8')
                        
            parent_jid = joint_info[-1]
            if parent_jid == -1: continue
            pjoint_info = pybullet.getJointInfo(self.bullet_robot, parent_jid)
            _pname = pjoint_info[12].decode('UTF-8')

            p_lid = _link_name_to_index[_pname]
            c_lid = _link_name_to_index[_name]
            if p_lid == -1 or c_lid == -1: continue

            if p_lid not in _neighbor_links:
                _neighbor_links[p_lid] = []

            _neighbor_links[p_lid].append(c_lid)


        for _id in range(pybullet.getNumJoints(self.bullet_robot)):
            joint_info = pybullet.getJointInfo(self.bullet_robot, _id)
            _pname = joint_info[12].decode('UTF-8')
            
            if _pname in disable_dict:
                _names = disable_dict[_pname]

                for _name in _names:
                    p_lid = _link_name_to_index[_pname]
                    c_lid = _link_name_to_index[_name]

                    if p_lid not in _neighbor_links:
                        _neighbor_links[p_lid] = []
                    _neighbor_links[p_lid].append(c_lid)

        # _neighbor_links[6].append(8)
        
        self.bullet_links_name = _link_name_to_index
        self.bullet_link_ids = _link_ids
        self.bullet_neighbor_links = _neighbor_links


    def set_base(self, pos):
        pybullet.resetBasePositionAndOrientation(
            self.bullet_robot, pos, [0,0,0,1]
        )
        
    def stop_render(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)

    def start_render(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)


    def add_collision(self, name, pos, quat_xyzw, size):
        cbox_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=np.array(size)/2, rgbaColor=[0,0,1,0.5])
        vbox_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=np.array(size)/2)

        box_id = pybullet.createMultiBody(baseMass=0,
            baseInertialFramePosition=[0,0,0],
            baseCollisionShapeIndex=cbox_id,
            baseVisualShapeIndex=vbox_id,
            basePosition=pos,
            baseOrientation=quat_xyzw)
        
        self.bullet_obstacles[name] = box_id

    def check_collision(self, joints):
        self.set_joints(joints)
        for obstacle_name in self.bullet_obstacles:
            obstacle_id = self.bullet_obstacles[obstacle_name]
            # closest_points = pybullet.getContactPoints(
            #     self.bullet_robot, obstacle_id)
            closest_points = pybullet.getClosestPoints(
                self.bullet_robot, obstacle_id, self.collision_min_dist)
            if closest_points is not None and len(closest_points) != 0:
                return True
        return False

    def check_self_collision(self, joints):
        
        '''
        Arguments: joints represents the current location of the robot
        Returns: A boolean where True means the arm has self-collision and false means that there are no collisions.
        '''
        self.set_joints(joints)
        for p_lid in self.bullet_link_ids:
            if p_lid not in self.bullet_neighbor_links: continue
            neighbor_lids = self.bullet_neighbor_links[p_lid]
            for c_lid in self.bullet_link_ids:
                if c_lid <= p_lid: continue
                if c_lid in neighbor_lids: continue
                # closest_points = pybullet.getContactPoints(
                #     self.bullet_robot, self.bullet_robot, p_lid, c_lid)
                closest_points = pybullet.getClosestPoints(
                    self.bullet_robot, self.bullet_robot, 0.01, p_lid, c_lid )
                if closest_points is not None and len(closest_points) != 0:
                    return True

        return False


    def set_joints(self, joints_values, joints_ids=None):
        if joints_ids is None:
            joints_ids = self.bullet_joints

        for j in range( len(joints_values) ):
            pybullet.resetJointState(
                self.bullet_robot,
                joints_ids[j],
                joints_values[j] )


    def ee(self, joints):
        '''
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''
        fk = self.forward_kinematics(joints)
        ee_frame = fk[-1,:,:]

        x, y, z = ee_frame[:-1,3]
        roll = np.arctan2(ee_frame[2,1], ee_frame[2,2])
        pitch = np.arcsin(-ee_frame[2,0])
        yaw = np.arctan2(ee_frame[1,0], ee_frame[0,0])

        return np.array([x, y, z, roll, pitch, yaw])

    def jacobian(self, joints):
        '''
        Calculate the jacobians analytically using your forward kinematics
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
        '''
        jacobian = np.zeros((6,self.num_dof))
        fk = self.forward_kinematics(joints)
        ee_pos = fk[-1,:3,3]

        for i in range(self.num_dof):
            joint_pos = fk[i,:3,3]
            joint_axis = fk[i,:3,2]
            jacobian[:3,i] = np.cross(joint_axis, (ee_pos - joint_pos)).T
            jacobian[3:,i] = joint_axis.T

        return jacobian

    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        joints = current_joints.copy()
        current_ee_pos = self.ee(joints)
        ee_error = desired_ee_pos - current_ee_pos
        alpha = 0.1

        while np.linalg.norm(ee_error) > 1e-3:
            jacob = self.jacobian(joints)
            joints += alpha * jacob.T.dot(ee_error.T)
            
            current_ee_pos = self.ee(joints)
            ee_error = desired_ee_pos - current_ee_pos

        return joints

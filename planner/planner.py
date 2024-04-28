import numpy as np
from planner.robot import Robot
from planner import RRT, RRTConnect, PRM, OBPRM


# import types
# def my_is_in_collision(self, joints):
#     if self.robot.check_self_collision(joints):
#         return True
#     for box in self.obstacles:
#         if self.robot.check_box_collision(joints, box):
#             return True
#     return False
# planner.is_in_collision = types.MethodType(my_is_in_collision, planner)




class Planner():
    def __init__(self, urdf_path, srdf_path, planner_name='rrtc') -> None:
        '''
        
        :param : 
            planner_name: str [ rrt | rrtc | prm | obprm ]  
        '''
       
        self.urdf_path = urdf_path
        self.srdf_path = srdf_path
        self.planner_name = planner_name

        self.robot = Robot()

        if planner_name == 'rrt':
            planner = RRT(self.robot, self.is_in_collision)
        elif planner_name == 'rrtc':
            planner = RRTConnect(self.robot, self.is_in_collision)
        elif planner_name == 'prm':
            planner = PRM(self.robot, self.is_in_collision)
        elif planner_name == 'obprm':
            planner = OBPRM(self.robot, self.is_in_collision)
        self.planner = planner

        self.obstacles = []

    def add_obstacles(self, shape):
        self.obstacles.append(shape)

    def clear_obstacles(self):
        # TODO ompl-style??
        pass

    def is_in_collision(self, joints):
        if self.robot.check_self_collision(joints):
            return True
        for box in self.obstacles:
            if self.robot.check_box_collision(joints, box):
                return True
        return False

    def ee_upright_constraint(self, q):
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
        desired_ee_rp = self.desired_ee_rp
        ee = self.robot.ee(q)
        err = np.sum((np.asarray(desired_ee_rp) - np.asarray(ee[3:5])) ** 2)
        grad = np.asarray([0, 0, 0, 2 * (ee[3] - desired_ee_rp[0]), 2 * (ee[4] - desired_ee_rp[1]), 0])
        return err, grad

    def get_plan_quality(self, plan):
        dist = 0
        for i in range(len(plan) - 1):
            dist += np.linalg.norm(np.array(plan[i+1]) - np.array(plan[i]))
        return dist

    def plan(self, joints_start, joints_target, constraint, args=None):
        planed_path = self.planner.plan(joints_start, joints_target, constraint, args)

        path_quality = self.get_plan_quality(planed_path)
        # print("Path quality: {}".format(path_quality))
        
        return planed_path

#!/usr/bin/python

# Completed by Charie Brady for CSC2630 Fall 2022 project.
# Extended using Florian Shkurti's A2 starter code.

import sys
import time
import pickle
import numpy as np
import random
import cv2
import environment
from math import sqrt
import copy
import threading

class State:

    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []
        if parent is None:
            self.start_cost = 0
        else:
            self.start_cost = self.euclidean_distance(parent) + parent.start_cost

    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def __str__(self):
        return f" [State: ({np.round(self.x, 2)}, {np.round(self.y,2)})  Cost: {np.round(self.start_cost)}  Children: {len(self.children)}  Parent: {self.parent}] "

    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x) ** 2 + (state.y - self.y) ** 2)

    def add_child(self, child):
        child.parent = self
        child.start_cost = self.start_cost + self.euclidean_distance(child)
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)


class RRTPlanner:
    """
    Applies the RRT algorithm on a given grid environment
    """

    def __init__(self, env, obs, obs_pos, obs_dim, radius):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.env = env
        self.img = env
        self.obs = obs

        # obstacle variables
        self.t = 0  # keeps track of time for moving obstacle
        self.obstacle_direction = -1
        self.obstacle_dimension = obs_dim
        self.obstacle_position = obs_pos
        self.collision_radius = None

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.env[:, :, 0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')
        self.occ_grid_o = self.obs[:, :, 0]
        self.occ_grid_o = (self.occ_grid_o == 0).astype('uint8')

        # tree variables
        self.plan_path = []
        self.plan_found = False
        self.start_state = None
        self.dest_state = None
        self.tree_nodes = None
        self.neighborhood_radius = radius +2
        self.dest_reached_radius = None
        self.max_steering_radius = None
        self.memory = []


    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[int(state.y - 10):int(state.y + 10), int(state.x - 10):int(state.x + 10)] == 1).all()

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        x = random.randint(0, len(self.env[1]))
        y = random.randint(0, len(self.env[0]))
        return State(x, y, None)

    def follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        assert (state is not None)

        curr_pt = state
        path = []

        t_end = time.time() + 0.1   # max path finding time

        while curr_pt is not None and time.time() < t_end and len(path) < 50:   # max path length
            path.append(curr_pt)
            curr_pt = curr_pt.parent

        return path[::-1]

    def find_neighbors(self, tree_nodes, state):
        neighbors = []
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist <= self.neighborhood_radius:
                neighbors.append(node)
        return neighbors

    def find_lowest_cost_state(self, neighbors, state):
        min_dist = float("Inf")
        lowest_cost_state = None
        for node in neighbors:
            dist = node.euclidean_distance(state) + node.start_cost
            if dist < min_dist:
                lowest_cost_state = node
                min_dist = dist
        return lowest_cost_state

    def rewire(self, s_new):
        neighbors = self.find_neighbors(self.tree_nodes, s_new)
        for node in neighbors:
            if node != self.start_state:
                dist = s_new.euclidean_distance(node)
                if ((s_new.start_cost + dist) < node.start_cost) and (self.path_is_obstacle_free(node, s_new)):
                    if node is not None and node.parent is not None:
                        self.remove_line(node, node.parent)
                        self.draw_line(node, s_new)
                        node.parent.children.remove(node)
                        s_new.add_child(node)

    def display_image(self, img):
        # update obstacle position in every new frame
        self.move_obstacle()
        # add tree
        self.img = img
        # add path if one is found
        if self.plan_found:
            self.draw_plan_path()
        # add start state
        cv2.circle(self.img, (int(self.start_state.x), int(self.start_state.y)), 6, (255, 255, 0), -1)
        # add dest state
        if len(self.memory) > 0:
            cv2.circle(self.img, (int(self.memory[0].x), int(self.memory[0].y)), int(dest_reached_radius), (0, 0, 255), 2)
        else:
            cv2.circle(self.img, (int(self.dest_state.x), int(self.dest_state.y)), int(dest_reached_radius), (0, 0, 255), 2)
        # add obstacle
        cv2.circle(self.img, (self.obstacle_position[0], self.obstacle_position[1]), int(self.obstacle_dimension), (0,0,255), -1)
        # show
        cv2.imshow('image', self.img)
        cv2.waitKey(10)

    def move_obstacle(self):
        # new image
        img = np.copy(self.env)

        # change direction every t = 50 movements
        if self.t % 100 == 0:
            self.obstacle_direction = self.obstacle_direction*-1
        self.t += 1

        #position change
        change = int(self.obstacle_dimension*2/10)

        # update position
        new_x = self.obstacle_position[0]
        new_y = self.obstacle_position[1] + change*self.obstacle_direction
        self.obstacle_position = [new_x, new_y]

        # draw new position
        cv2.circle(img, (self.obstacle_position[0], self.obstacle_position[1]), int(self.obstacle_dimension+self.collision_radius), (255,255,255), -1)
        cv2.circle(img, (self.obstacle_position[0], self.obstacle_position[1]), int(self.obstacle_dimension), (0,0,255), -1)
        self.obs = img

        # update occupancy grid
        self.occ_grid_o = self.obs[:, :, 0]
        self.occ_grid_o = (self.occ_grid_o == 0).astype('uint8')

    def draw_line(self, start_state, end_state, colour='blue'):
        # blue is default
        global BGR
        if colour == 'blue':
            BGR = (int(255), (0), (0))
        if colour == 'white':
            BGR = (int(255), int(255), int(255))
        if colour == 'red':
            BGR = (int(0), int(0), int(255))
        if colour == "yellow":
            BGR = (int(255), int(255), int(0))
        if colour == "green":
            BGR = (int(255), int(255), int(0))
        cv2.line(self.img, (int(start_state.x), int(start_state.y)), (int(end_state.x), int(end_state.y)), BGR)
        cv2.circle(self.img, (int(start_state.x), int(start_state.y)), int(2), BGR, -1)
        cv2.circle(self.img, (int(end_state.x), int(end_state.y)), int(2), BGR, -1)
        # add dest state
        if len(self.memory) > 0:
            cv2.circle(self.img, (int(self.memory[0].x), int(self.memory[0].y)), int(dest_reached_radius), (0, 0, 255), 2)
        else:
            cv2.circle(self.img, (int(self.dest_state.x), int(self.dest_state.y)), int(dest_reached_radius), (0, 0, 255), 2)

    def remove_line(self, start_state, end_state):
        cv2.line(self.img, (int(start_state.x), int(start_state.y)), (int(end_state.x), int(end_state.y)), (0, 0, 0))
        cv2.circle(self.img, (int(start_state.x), int(start_state.y)), int(2), (255, 0, 0), -1)
        cv2.circle(self.img, (int(end_state.x), int(end_state.y)), int(2), (255, 0, 0), -1)

    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist and dist != 0:
                closest_state = node
                min_dist = dist
        return closest_state

    def update_dest_state(self, new_dest_state):

        if new_dest_state is self.dest_state:
            self.plan_found = True
            return
        if self.state_is_free(new_dest_state):
            # update destination state
            self.dest_state = new_dest_state
            # search for closest state to new dest
            s_nearest = self.find_closest_state(self.tree_nodes, self.dest_state)
            if s_nearest.euclidean_distance(self.dest_state) <= self.dest_reached_radius and self.path_is_obstacle_free(s_nearest, self.dest_state):
                self.tree_nodes.add(self.dest_state)
                self.plan_found = True
                s_nearest.add_child(self.dest_state)
            else:
                # update plan_found variable to false
                self.plan_found = False
                self.plan_path = []

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:
        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y
        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.
        """
        x = 0
        y = 0

        dist = s_nearest.euclidean_distance(s_rand)

        if max_radius >= dist:
            x = s_rand.x
            y = s_rand.y
        else:
            y = ((s_rand.y - s_nearest.y) / dist) * max_radius + s_nearest.y
            x = ((s_rand.x - s_nearest.x) / dist) * max_radius + s_nearest.x

        s_new = State(x, y, s_nearest)

        return s_new

    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))
        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10
        for i in range(max_checks):
            scale_x = (i / max_checks) * (s_to.x - s_from.x)
            scale_y = (i / max_checks) * (s_to.y - s_from.y)
            x_temp = int(s_from.x + scale_x)
            y_temp = int(s_from.y + scale_y)
            temp = State(x_temp, y_temp, None)
            if not self.state_is_free(temp):
                return False

        # Otherwise the line is free, so return true
        return True

    def update_tree_costs(self):

        visited = [self.start_state]
        stack = [self.start_state]

        self.start_state.start_cost = 0
        while stack:
            if len(stack) == 0:
                break
            node = stack.pop(0)
            if node not in visited:
                visited.append(node)
            for child in node.children:
                child.start_cost = node.start_cost + child.euclidean_distance(node)
                if child not in visited:
                    stack.append(child)


    def rewire_whole_tree_s(self):

        visited = [self.start_state]
        stack = [self.start_state]

        self.start_state.start_cost = 0
        self.rewire(self.start_state)

        while stack:
            if len(stack) == 0:
                break
            node = stack.pop(0)
            visited.append(node)
            if node not in visited:
                visited.append(node)
            for child in node.children:
                child.start_cost = node.start_cost + child.euclidean_distance(node)      # update costs before rewiring
                if child not in visited:
                    stack.append(child)
                self.rewire(child)

        img = self.draw_tree()
        self.display_image(img)

        return visited

    def rewire_whole_tree(self):
        self.update_tree_costs()

        for node in self.tree_nodes:
            self.rewire(node)

        self.display_image(self.draw_tree())

    def draw_tree(self, colour=None):
        BGR = (int(255), int(0), int(0))
        if colour == 'white':
            BGR = (int(255), int(255), int(255))
        copy = np.copy(self.env)
        # plot tree
        for state in self.tree_nodes:
            if state.parent is not None:
                cv2.line(copy, (int(state.x), int(state.y)), (int(state.parent.x), int(state.parent.y)), BGR)
                cv2.circle(copy, (int(state.x), int(state.y)), int(2), BGR, -1)
                cv2.circle(copy, (int(state.parent.x), int(state.parent.y)), int(2), BGR, -1)
        return copy

    def resample_root(self, child, new_root):    # no longer used
        sample = None

        while sample == None:
            min_x = np.minimum(child.x, new_root.x)
            max_x = np.maximum(child.x, new_root.x)
            min_y = np.minimum(child.y, new_root.y)
            max_y = np.maximum(child.y, new_root.y)
            x = random.randint(int(min_x - 1), int(max_x + 1))
            y = random.randint(int(min_y - 1), int(max_y + 1))

            s_rand = State(x, y, None)

            dist_c = child.euclidean_distance(s_rand)
            dist_r = new_root.euclidean_distance(s_rand)

            if dist_c <= (self.neighborhood_radius + 3) and dist_r <= (self.neighborhood_radius + 3) \
                    and self.state_is_free(s_rand) and self.path_is_obstacle_free(child, s_rand) \
                    and self.path_is_obstacle_free(s_rand, new_root):
                sample = s_rand

        return sample

    def rewire_root(self, new_root):

        if new_root in self.start_state.children:
            self.start_state.children.remove(new_root)

        new_root.add_child(self.start_state)
        new_root.start_cost = 0
        new_root.parent = None
        self.start_state = new_root

        self.display_image(self.draw_tree())


    def return_recursed_tree(self, state, list): # not used
        if state == self.start_state:
            visited = [state]
            for child in self.start_state.children:
                self.return_recursed_tree(child, visited)
            return visited
        elif state.children == []:
            list.append(state)
            return
        else:
            list.append(state)
            for child in state.children:
                self.return_recursed_tree(child, list)

    def draw_plan_path(self):

        if self.plan_found and self.start_state != self.dest_state:
            self.plan_path = self.follow_parent_pointers(self.dest_state)

        # draw new path (red)
            if self.plan_path is not None:
                if 1 < len(self.plan_path) < 50:
                    for i in range(len(self.plan_path) - 1):
                        pt0 = (int(self.plan_path[i].x), int(self.plan_path[i].y))
                        pt1 = (int(self.plan_path[i + 1].x), int(self.plan_path[i + 1].y))
                        cv2.line(self.img, pt0, pt1, (0, 0, 255))  # red
                    # draw target area
                    if len(self.memory) > 0:
                        cv2.circle(self.img, (int(self.memory[0].x), int(self.memory[0].y)), int(dest_reached_radius), (0, 0, 255), 2)
                    else:
                        cv2.circle(self.img, (int(self.dest_state.x), int(self.dest_state.y)), int(dest_reached_radius), (0, 0, 255), 2)


    def expand_tree(self, s_nearest=None):
        s_rand = self.sample_state()
        if s_nearest is None:
            s_nearest = self.find_closest_state(self.tree_nodes, s_rand)
        else:
            s_nearest = self.start_state
        s_new = self.steer_towards(s_nearest, s_rand, self.max_steering_radius)
        neighbors = self.find_neighbors(self.tree_nodes, s_new)
        s_lowest_cost = self.find_lowest_cost_state(neighbors, s_new)
        dist = s_lowest_cost.euclidean_distance(s_new)
        if self.path_is_obstacle_free(s_lowest_cost, s_new) and dist > self.max_steering_radius/2:
            # add new state to lowest cost state
            s_lowest_cost.add_child(s_new)
            # rewire neighborhood of new state
            self.rewire(s_new)  # (neighbors, self.start_state)
            # add new state to tree
            self.tree_nodes.add(s_new)
            return s_new
        else:
            return None

    def build_initial_tree(self, size):
        for i in range(size):
            s_new = None
            while s_new is None:
                s_new = self.expand_tree()

    def view_destinations(self):
        copy = np.copy(self.env)
        # plot tree
        for state in self.tree_nodes:
            cv2.circle(copy, (int(state.x), int(state.y)), int(3), (0, 0, 255), -1)
            if len(state.children) == 0:
                cv2.circle(copy, (int(state.x), int(state.y)), int(3), (0, 255, 0), -1)
        cv2.imshow('copy', copy)
        cv2.waitKey(10)
        return copy

    def state_is_obstacle_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        assert(state is not None)
        return (self.occ_grid_o[int(state.y - 1):int(state.y + 1), int(state.x - 1):int(state.x + 1)] == 1).all()

    def path_is_dyn_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))
        if not (self.state_is_obstacle_free(s_to)):
            return False

        max_checks = 10
        for i in range(max_checks):
            scale_x = (i / max_checks) * (s_to.x - s_from.x)
            scale_y = (i / max_checks) * (s_to.y - s_from.y)
            x_temp = int(s_from.x + scale_x)
            y_temp = int(s_from.y + scale_y)
            temp = State(x_temp, y_temp, None)
            if not self.state_is_obstacle_free(temp):
                return False

        # Otherwise the line is free, so return true
        return True

    def plan_is_free(self, plan):
        for i in range(len(plan)-1):
            if self.path_is_dyn_obstacle_free(plan[i], plan[i+1]) is False:
                return False
        return True

    def find_all_free_dest_plans(self, dest_state):
        plans = []
        neighbors = self.find_neighbors(self.tree_nodes, dest_state)
        for neighbor in neighbors:
                try:
                    plan = self.follow_parent_pointers(neighbor)
                    if neighbor in plan and self.plan_is_free(plan):
                        plans.append(plan)
                except:
                    continue
        return plans

    def find_best_free_dest_plan(self, dest_state):
        plans = self.find_all_free_dest_plans(dest_state)
        min = float("Inf")
        closest = None
        for plan in plans:
            dist = plan[-1].euclidean_distance(dest_state)
            by_obs = plan[-1].euclidean_distance(State(self.obstacle_position[0], self.obstacle_position[1], None))
            if dist < min and by_obs > self.collision_radius:
                min = dist
                closest = plan
        return closest

    def draw_alt_plans(self, plans):
        for plan in plans:
            for i in range(len(plan) - 1):
                pt0 = (int(plan[i].x), int(plan[i].y))
                pt1 = (int(plan[i + 1].x), int(plan[i + 1].y))
                cv2.line(self.img, pt0, pt1, (255,255,255))
            cv2.imshow('image', self.img)
            cv2.waitKey(10)
            time.sleep(0.1)  # show for demo


    def update_plan(self, plan):
        if not self.memory:
            self.memory.append(copy.copy(self.dest_state))
        self.plan_path = plan
        self.draw_alt_plans([self.plan_path])
        self.update_dest_state(self.plan_path[-1])
        self.plan_found = True


    def run(self, start_state, dest_state, max_num_steps, size, max_nodes, max_steering_radius, dest_reached_radius, collision_radius, k_max):

        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        self.start_state = start_state
        self.dest_state = dest_state
        self.dest_reached_radius = dest_reached_radius
        self.max_steering_radius = max_steering_radius
        self.collision_radius = collision_radius

        # The set containing the nodes of the tree
        self.tree_nodes = set()
        self.tree_nodes.add(self.start_state)

        # image to be used to display the tree
        self.img = np.copy(self.env)

        # initialize image: add start and dest states and mouse callback
        cv2.circle(self.img, (int(self.start_state.x), int(self.start_state.y)), 6, (255, 255, 0), -1)
        cv2.circle(self.img, (int(self.dest_state.x), int(self.dest_state.y)), self.dest_reached_radius, (0, 0, 255), 2)
        cv2.imshow('image', self.img)
        cv2.setMouseCallback('image', mouse_click)

        self.build_initial_tree(size)

        for step in range(max_num_steps):

            #self.view_destinations()

            if step < max_nodes or step % 20 == 0:    # randomly add a node and rewire every 20th iteration

                s_new = None
                while s_new is None:
                    s_new = self.expand_tree()

                # if the new state reaches the destination, add dest_state as child
                if s_new.euclidean_distance(self.dest_state) < self.dest_reached_radius:
                    if not self.plan_found:
                        self.tree_nodes.add(self.dest_state)
                        self.plan_found = True
                    s_new.add_child(self.dest_state)

            # if a plan_path has been found, update the img with the new path
            # check if plan_path is collision free. replan if not
            if self.plan_found and self.state_is_obstacle_free(self.start_state):
                self.draw_plan_path()

                # for each state in the plan, move along it towards the destination
                for state in self.plan_path:
                    if state != self.start_state:
                        plan_is_free = self.plan_is_free(self.plan_path)
                        # find alternative plan if plan within collision radius
                        if not plan_is_free:
                            plan = self.find_best_free_dest_plan(self.dest_state)
                            if plan is not None:
                                self.update_plan(plan)
                                break
                            else:
                                for k in range(k_max, 2, -1):
                                    if len(self.plan_path) > k:
                                        plan = self.find_best_free_dest_plan(self.plan_path[k - 1])
                                        if plan is not None:
                                            self.update_plan(plan)
                                            break
                                self.display_image(self.draw_tree())
                                break
                        else:
                            self.rewire_root(state)
                            break

                if self.dest_state == self.start_state:
                    if self.memory == [] or self.dest_state in self.memory:
                        self.rewire_whole_tree()
                        self.plan_found = False
                        self.plan_path = []
                        self.memory = []
                    else:
                        self.update_dest_state(self.memory.pop())
                        self.rewire_whole_tree()
            else:
                self.display_image(self.draw_tree())


if __name__ == "__main__":
    loaded_env = None
    if len(sys.argv) > 1:
        pkl_file = open(sys.argv[1], 'rb')
        loaded_env = pickle.load(pkl_file)
        pkl_file.close()

    e0 = environment.Environment(dim=550)

    if loaded_env is not None:
        world = e0.invert_environment(loaded_env)
    else:
        #world = e0.build_environment(num_blocks=6, num_walls=3)   # generates random environment
        world = e0.build_default()  # generates default environment

    e1 = environment.Environment(dim=550)
    obstacle = e1.build_dynamic_obstacle()
    obs_pos = e1.get_obstacle_position()
    obs_dim = e1.get_obstacle_dim()

    start_state = State(50, 50, None)
    dest_state = State(500, 500, None)

    max_num_steps = 5000  # max number of nodes to be added to the tree
    size = 600 # initial size of tree. 0 will initialize an empty tree that builds to max_nodes size
    max_nodes = 0  # max additional nodes added to the tree
    max_steering_radius = 50  # pixels
    dest_reached_radius = 15  # pixels
    collision_radius = 25
    k_max = 4

    def mouse_click(event, x, y, flags, params):
        # Left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            rrt.memory = []
            rrt.plan_found = False
            rrt.plan_path = []
            rrt.update_dest_state(State(x, y, None))

    rrt = RRTPlanner(world, obstacle, obs_pos, obs_dim, max_steering_radius)

    rrt.run(start_state,
            dest_state,
            max_num_steps,
            size,
            max_nodes,
            max_steering_radius,
            dest_reached_radius,
            collision_radius,
            k_max)
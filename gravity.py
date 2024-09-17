import json
import math
import random
import time
import tkinter as tk
from os import listdir
from os.path import isfile, join
from queue import PriorityQueue
from tkinter import *

import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure

# Gravity Code
# All of the code is run through the UserInterface class
# Individual simulations are run by the Simulation class
# The rest of the class's and functions are used by those two classes

# Global variables determining broad information
G = 1
font = "Times"
models = ["Interval", "Dynamic Interval", "Velocity Average", "Simultaneous Calculation"]
model_descriptions = {
    "Interval": "This is the basic model where every time step the gravitational attraction is calculated and the position updated.",
    "Dynamic Interval": "In this model a body is updated in smaller time intervals when it is passing close to another body.",
    "Velocity Average": "In this model instead of calculating the gravitational interaction at a single position.\nIt is calculated at the average position between current and next position.",
    "Simultaneous Calculation": "This is mathmatically identical to interval, but calculates all interaction in the same step for better performance."
}


# Helper Functions and Classes

# Returns the distance between two points
def dist(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


# Returns the distance between two points squared
def dist_squared(pos1, pos2):
    return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2


# Returns distance in the x direction between two points
def component_x(pos1, pos2):
    return (pos2[0] - pos1[0]) / dist(pos1, pos2)


# Returns distance in the y direction between two points
def component_y(pos1, pos2):
    return (pos2[1] - pos1[1]) / dist(pos1, pos2)

# handles a collision between 2 planet objects
def collide(planets, p1, p2):
    if p1 in planets and p2 in planets:
        p1.pos[0] = (p1.pos[0] * p1.mass + p2.pos[0] * p2.mass) / (p1.mass + p2.mass)
        p1.pos[1] = (p1.pos[1] * p1.mass + p2.pos[1] * p2.mass) / (p1.mass + p2.mass)
        p1.vel[0] = (p1.vel[0] * p1.mass + p2.vel[0] * p2.mass) / (p1.mass + p2.mass)
        p1.vel[1] = (p1.vel[1] * p1.mass + p2.vel[1] * p2.mass) / (p1.mass + p2.mass)
        p1.mass += p2.mass
        p1.radius = ((p1.radius ** 3) + (p2.radius ** 3)) ** (1 / 3)
        planets.remove(p2)


# A tkinter scale object where the scale increases by different factors at differenct points
# The code was modified from code taken from scotty3785 posted on this page:
# https://stackoverflow.com/questions/69175812/in-a-tkinter-slider-can-you-vary-the-range-from-being-linear-to-exponential-for
class NonLinearScale(tk.Scale):
    def __init__(self, parent, **kwargs):
        tk.Scale.__init__(self, parent, **kwargs)
        self.var = tk.DoubleVar()
        self['showvalue'] = 0
        self['label'] = 0
        self['command'] = self.update
        self['variable'] = self.var

    def update(self, event):
        self.config(label=self.value)

    @property
    def value(self):
        if int(self.var.get()) < 20:
            return str(int(self.var.get()))
        elif int(self.var.get()) < 50:
            return str(5 * (int(self.var.get()) - 16))
        else:
            return str(10 * (int(self.var.get()) - 35))


# A tkinter scale object where the scale increases by an exponential factor.
# The code was modified from code posted by scotty3785 on this page:
# https://stackoverflow.com/questions/69175812/in-a-tkinter-slider-can-you-vary-the-range-from-being-linear-to-exponential-for
class LogScale(tk.Scale):
    def __init__(self, parent, **kwargs):
        tk.Scale.__init__(self, parent, **kwargs)
        self.var = tk.DoubleVar()
        # self.config[showvalue] = 0
        self['showvalue'] = 0
        self['label'] = 0
        self['command'] = self.update
        self['variable'] = self.var

    def update(self, event):
        self.config(label=self.value)

    @property
    def value(self):
        return str(10 ** float(self.var.get()))

    def get(self):
        return 10 ** float(self.var.get())


# A tkinter object for displaying text when hovering over another widget
# The code was modified from code posted by squareRoot17 on this page:
# https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.label = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text, add_y):
        if type(text) == StringVar:
            text = text.get()
        self.text = text
        if not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx()
        y = y + cy + self.widget.winfo_rooty() - 40 + add_y
        global window
        self.label = Label(window, text=self.text, justify=LEFT,
                           background="#ffffe0", relief=SOLID, borderwidth=1,
                           font=("tahoma", "8", "normal"))
        self.label.place(x=x, y=y)

    def hidetip(self):
        self.label.destroy()


# Function for managing the ToolTip class, also posted by squareRoot17
def CreateToolTip(widget, text, add_y=0):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text, add_y)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


# Planet class, each planet object represents a single object in the simulation.
class Planet:
    pos = np.array([0.0, 0.0])
    vel = np.array([0.0, 0.0])
    mass = 1
    radius = 1

    def __init__(self, pos, vel, mass, radius, path_color="green", color="black"):
        global planet_number
        self.planet_number = planet_number
        planet_number += 1
        self.last_pos = np.array([0.0, 0.0])
        self.last_pos = np.copy(pos)
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = radius
        self.next_update = 0
        self.path_color = path_color
        self.color = color

    # Updates the planet position and velocity by calculating its gravitational interaction between other planets.
    def update_planet(self, planets, model):
        maxA = 0
        min_dis = -1
        acceleration = np.array([0.0, 0.0])
        global collisions
        global position_average_modifier
        position_average_modifier = 1
        collision = False
        calculation_position = self.pos.copy()
        # if model == "Position Average":
        #     calculation_position += self.vel * position_average_modifier
        for p2 in planets:
            if self != p2:
                if min_dis == -1 or min_dis > dist(self.pos, p2.pos):
                    if collisions and dist(self.pos, p2.pos) < self.radius + p2.radius:
                        collide(planets, self, p2)
                        collision = True
                    else:
                        min_dis = dist(calculation_position, p2.pos)
                if not (collision == True):
                    p2_calculation_position = p2.pos.copy()
                    # if model == "Position Average":
                    #     p2_calculation_position += p2.vel * position_average_modifier
                    acceleration[0] += component_x(calculation_position,
                                                  p2_calculation_position) * p2.mass / dist_squared(calculation_position,
                                                                                                   p2_calculation_position)
                    acceleration[1] += component_y(calculation_position,
                                                  p2_calculation_position) * p2.mass / dist_squared(calculation_position,
                                                                                                   p2_calculation_position)
                    collision = False
        velocity = self.vel + acceleration
        global dynamic_interval_scale
        if model == "Dynamic Interval" and min_dis > 0:

            dynamic_interval_modifier = float(dynamic_interval_scale.get())
            if dist([0, 0], velocity) * dynamic_interval_modifier > min_dis:
                next_timestep = (1 * min_dis) / (dynamic_interval_modifier * dist([0, 0], velocity))

            else:
                next_timestep = 1
        else:
            next_timestep = 1
        update_modifier = 1
        if model == "Velocity Average":
            update_modifier = 0.75
        self.next_update += next_timestep
        self.last_pos = self.pos

        self.vel += update_modifier * next_timestep * acceleration
        self.pos += next_timestep * self.vel
        self.vel += (1 - update_modifier) * next_timestep * acceleration

    # Determines planet order in priority queues so planets with sooner next_updates are prioritized
    def __gt__(self, other):
        if isinstance(other, Planet):
            return self.next_update > other.next_update

    def __str__(self):
        return "Pos: " + str(self.pos) + "  Vel: " + str(self.vel) + "  Mass: " + str(self.mass) + \
            "  Radius: " + str(self.radius) + "  Next Update: " + str(self.next_update)


# SimulationInstance class, represents one instance of a simulation
class SimulationInstance:
    def __init__(self, planets, model, user_interface):
        self.user_interface = user_interface
        self.running = False
        self.current_time = 0
        self.planets = planets
        self.planet_queue = PriorityQueue()
        for p in planets:
            self.planet_queue.put(p)
        self.model = model
        self.recalculate_com()
        self.times_list = [0]
        self.k_energy_list = []
        self.gp_energy_list = []
        if len(planets) == 2:
            self.two_planet_start = True
            self.starting_paths = self.calculate_actual_paths()
        else:
            self.two_planet_start = False
            self.starting_paths = None
        self.variances_list = [[], []]
        self.number_planets = len(planets)


    # calculates the center of mass for the planet system
    def recalculate_com(self):
        self.com = np.array([0.0, 0.0])
        totalMass = 0
        for p in self.planets:
            self.com += p.pos * p.mass
            totalMass += p.mass
        self.com = self.com / totalMass

    # updates all the planets through one timestep
    def update(self):
        if self.model == "Simultaneous Calculation":
            position_list = []
            velocity_list = []
            masses_list = []
            radius_list = []
            for p in self.planets:
                position_list.append(p.pos)
                velocity_list.append(p.vel)
                masses_list.append(p.mass)
                radius_list.append(p.radius)
            position_array = np.array(position_list)
            mass_array = np.array(masses_list)
            velocity_array = np.array(velocity_list)
            radius_array =  np.array(radius_list)
            horizontal_position_array = position_array.reshape((len(self.planets), 1, 2))
            verticle_position_array = position_array.reshape((1, len(self.planets), 2))
            distance_vectors = -1*(horizontal_position_array - verticle_position_array)
            distances = np.sqrt(np.sum(np.square(distance_vectors),axis=2)).reshape((len(self.planets), len(self.planets), 1))
            collision_distances = np.add(radius_array.reshape((len(self.planets), 1)), radius_array.reshape((1, len(self.planets))))
            _distances = np.reshape(distances,(len(self.planets),(len(self.planets))))
            collision = np.greater(collision_distances, _distances, out =np.full((len(self.planets), len(self.planets)), False),  where=_distances!=0)
            colliding_indeces = np.where(np.any(collision, axis=1))[0]
            a = distance_vectors*mass_array.reshape((1, len(self.planets), 1))
            b = np.power(distances,3)
            accelerations = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            accelerations = np.sum(accelerations,axis = 1)
            velocity_array += accelerations
            position_array += velocity_array
            for i in range(len(self.planets)):
                self.planets[i].pos = position_array[i]
                self.planets[i].vel = velocity_array[i]
            planets_to_collide = []
            for i in colliding_indeces:
                for j in range(i, len(collision[i])):
                    if collision[i][j]:
                        planets_to_collide.append([self.planets[i],self.planets[j]])
            for p in planets_to_collide:
                global collisions
                if collisions:
                    collide(self.planets, p[0], p[1])
            self.recalculate_com()


        else:
            p = self.planet_queue.get()
            while p.next_update < self.current_time + 1:
                if p in self.planets:
                    # timestep = p.next_update - current_time
                    # current_time = p.next_update
                    p.update_planet(self.planets, self.model)
                    self.recalculate_com()
                    self.planet_queue.put(p)
                p = self.planet_queue.get()
            self.planet_queue.put(p)
        self.current_time += 1
        self.times_list.append(self.current_time)

    # calculated the path the planets should follow under gravitational interaction
    # method only works if there are exactly two planets
    def calculate_actual_paths(self):
        if len(self.planets) != 2:
            return "ERROR: can't calculate path, there must be exactly 2 planets"
        p1 = self.planets[0]
        p2 = self.planets[1]
        com_vel = (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
        r1 = dist(self.com, p1.pos)
        r2 = dist(self.com, p2.pos)
        tangent1 = np.array([(p1.pos - self.com)[1], -1 * (p1.pos - self.com)[0]])
        tangent2 = np.array([(p2.pos - self.com)[1], -1 * (p2.pos - self.com)[0]])
        component_v1 = np.dot(p1.vel - com_vel, tangent1) / np.linalg.norm(tangent1)
        component_v2 = np.dot(p2.vel - com_vel, tangent2) / np.linalg.norm(tangent2)
        m = p1.mass * p2.mass / (p1.mass + p2.mass)
        k = G * p1.mass * p2.mass
        L = p1.mass * component_v1 * r1 + p2.mass * component_v2 * r2
        E = 0.5 * p1.mass * dist(p1.vel - com_vel, np.zeros(2)) ** 2 \
            + 0.5 * p2.mass * dist(p2.vel - com_vel, np.zeros(2)) ** 2 - k / (r1 + r2)
        ecc = np.sqrt(1 + (2 * E * L ** 2) / (m * k ** 2))

        def calculate_planet_path(planet, r):
            sin = 1
            going_out = np.dot(planet.pos - self.com, planet.vel - com_vel) < 0
            left_side = np.degrees(np.arctan2(planet.vel[1] - com_vel[1], planet.vel[0] - com_vel[0])
                                   - np.arctan2(planet.pos[1] - self.com[1], planet.pos[0] - self.com[0])) % 360 < 180
            if (going_out and not left_side) or (not going_out and left_side):
                sin = -1
            inner_cos = (1 / ecc) * ((L ** 2 / (k * planet.mass * r)) - 1)
            if inner_cos < -1:
                inner_cos = -1
            if inner_cos > 1:
                inner_cos = 1

            theta0 = sin * np.arccos(inner_cos) \
                     + np.arctan2((planet.pos[1] - self.com[1]), ((planet.pos[0] - self.com[0])))
            points = []
            if ecc < 1:
                assymptope_angle = 181
            else:
                assymptope_angle = 179 - math.floor(np.degrees(np.arccos(1 / ecc)))
            for theta in range(-1 * assymptope_angle, assymptope_angle + 1):
                theta = np.deg2rad(theta)
                radius = (L ** 2 / (planet.mass * k)) / (1 + ecc * np.cos(theta))
                points.append(radius * np.cos(theta + theta0))
                points.append(radius * np.sin(theta + theta0))

            return points

        p1_path = calculate_planet_path(p1, r1)
        p2_path = calculate_planet_path(p2, r2)
        return [p1_path, p2_path]

    # records the kinetic and gravitational potential energy of the planet system
    def record_energys(self):
        def gp_energy(planets):
            gp_energy = 0
            for i in range(1, len(planets)):
                for j in range(0, i):
                    gp_energy += planets[i].mass * planets[j].mass / (planets[i].radius + planets[j].radius) - planets[
                        i].mass * planets[j].mass / dist(planets[i].pos, planets[j].pos)

            return gp_energy

        def k_energy(planets):
            k_energy = 0
            for p in planets:
                k_energy += 0.5 * p.mass * dist_squared(p.vel, [0, 0])
            return k_energy

        self.k_energy_list.append(k_energy(self.planets))
        self.gp_energy_list.append(gp_energy(self.planets))

    # records the difference betweeneach planets position and path they should be on.
    # method only works if there were exactly two planets to start
    def record_variance(self):
        if self.two_planet_start == False:
            return "ERROR: can't record variance, there must be exactly 2 planets"
        if len(self.planets) < 2:
            self.variances_list[0].append(0)
            self.variances_list[1].append(0)
            return "ERROR: can't record variance, there must be exactly 2 planets"

        def variance_from_path(planet, path):
            def dist_line_point(x1, y1, x2, y2, point):
                x0 = point[0]
                y0 = point[1]
                # equation taken from wikipedia
                return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
                    (y2 - y1) ** 2 + (x2 - x1) ** 2)

            min = -1
            min_index = -1
            for i in range(0, len(path), 2):
                distance = dist(np.array([path[i], path[i + 1]]), planet.pos - self.com)
                if min == -1 or distance < min:
                    min = distance
                    min_index = i
            next_index = min_index + 2
            prev_index = min_index - 2
            if min_index == len(path) - 2:
                next_index = 0
            if min_index == 0:
                prev_index = len(path) - 2
            if dist(np.array([path[next_index], path[next_index + 1]]), planet.pos - self.com) < dist(
                    np.array([path[prev_index], path[prev_index + 1]]), planet.pos - self.com):
                return dist_line_point(path[min_index], path[min_index + 1], path[next_index], path[next_index + 1],
                                       planet.pos - self.com)
            else:
                return dist_line_point(path[prev_index], path[prev_index + 1], path[min_index], path[min_index + 1],
                                       planet.pos - self.com)

        for i in range(2):
            self.variances_list[i].append(variance_from_path(self.planets[i], self.starting_paths[i]))

    # checks if an end condition has been met by the current simulation
    def end_condition_met(self):
        if self.user_interface.draw_visuals == True:
            return False
        try:
            self.user_interface.end_condition
        except AttributeError:
            return False
        if self.user_interface.end_condition.get() == "First Collision" or self.user_interface.end_condition.get() == "Time or First Collision":
            if self.number_planets > len(self.planets):
                return True
        if self.user_interface.end_condition.get() == "Time" or self.user_interface.end_condition.get() == "Time or First Collision":
            if self.current_time > float(self.user_interface.end_time_scale.get()):
                return True
        return False

    # method that handles setting up and running the simulation
    def run_simulation(self):
        self.running = True
        while self.running:
            self.record_energys()
            self.record_variance()
            if self.user_interface.draw_visuals:
                if self.user_interface.graph_energy and self.current_time % (self.user_interface.graph_update_scale.get()) == 0:
                    self.user_interface.plot_energy(self.times_list, self.k_energy_list, self.gp_energy_list)
                if self.user_interface.graph_actual_variance and self.current_time % (
                        self.user_interface.graph_update_scale.get()) == 0 and self.two_planet_start:
                    self.user_interface.plot_variance(self.times_list, self.variances_list)
                self.user_interface.draw(self)
            self.update()
            if self.end_condition_met():
                self.running = False
            if self.user_interface.draw_visuals:
                time.sleep(self.user_interface.timestep_length_scale.get())


# UserInterface class, handles all the user inputs and keeps track of the simulations that are running
class UserInterface:
    def __init__(self, window):
        self.window = window
        self.window.geometry("1600x900")
        self.window.configure(background="lightblue")
        self.simulations = []
        self.simulations_running = False

        self.create()
        self.window.mainloop()

    # Creates and stores all the widgets for the user interface
    def create(self):
        # Creates the title widget
        def create_title():
            title_frame = tk.Frame(self.window, highlightbackground="black", width=1580, height=55,
                                   highlightthickness=2)
            title_frame.grid(row=0, column=0, columnspan=3, pady=8, padx=8, sticky=(N, S, E, W))
            title = tk.Label(title_frame, text="2D N-Body Gravitational Simulator", font=(font, 35))
            title.place(relx=0.5, rely=0.5, anchor=CENTER)
            # self.window.columnconfigure(1, weight=1)
            # self.window.rowconfigure(1, weight=1)

        # Creates the widget that contains all of the controls for the simulation
        def create_controls():
            start_condition = StringVar()
            controls_font_size = 12
            scale_width = 10

            # method that is called when a parrameter relating to how the visuals should be drawn is changed.
            # it updates the screen rules updated variable,.
            # room to expand it if other requirements for updating the screen rules appear
            def _screen_rules_updated(*args):
                self.screen_rules_updated = True

            # Creates the Widget for the general controls
            def create_general_controls(control_frame):
                general_font_size = controls_font_size

                def create_visuals_button(draw_control_frame):
                    self.draw_visuals = True
                    draw_visuals_frame = tk.Frame(draw_control_frame)

                    def draw_visuals_button_clicked():
                        if len(self.simulations) != 0 and self.simulations_running:
                            return
                        if self.draw_visuals:
                            draw_visuals_frame.draw_visuals_button.config(highlightbackground="red",
                                                                          text="Not Showing Simulation")
                            self.draw_visuals = False
                            self.draw_control_frame.destroy()
                            create_multiple_simulation_controls(control_frame)
                            self.canvas.create_rectangle(-10, 370, 820, 430, fill="grey")
                            self.canvas.create_text(400, 400, text="Not Showing Simulation", fill="black", font=(font, 40))
                        else:
                            draw_visuals_frame.draw_visuals_button.config(highlightbackground="green",
                                                                          text="Showing Simulation")
                            self.draw_visuals = True
                            self.multiple_simulation_control_frame.destroy()
                            create_draw_controls(control_frame)
                            self.canvas.delete("all")

                    draw_visuals_frame.draw_visuals_button = tk.Button(draw_visuals_frame, text="Showing Simulation",
                                                                       highlightbackground="green",
                                                                       command=draw_visuals_button_clicked,
                                                                       font=(font, general_font_size), width=20)

                    draw_visuals_label = tk.Label(draw_visuals_frame, text="Visuals: ",
                                                  font=(font, general_font_size))
                    draw_visuals_label.pack(side=LEFT)
                    draw_visuals_frame.draw_visuals_button.pack(side=LEFT)
                    draw_visuals_frame.pack(side=TOP, anchor=NW)

                def create_collisions_button(general_frame):
                    global collisions
                    collisions = True
                    collisions_frame = tk.Frame(general_frame)

                    def collisions_button_clicked():
                        global collisions
                        if collisions:
                            collisions_frame.collisions_button.config(highlightbackground="red", text="Off")
                            collisions = False
                        else:
                            collisions_frame.collisions_button.config(highlightbackground="green", text="On")
                            collisions = True

                    collisions_frame.collisions_button = tk.Button(collisions_frame, text="On",
                                                                   highlightbackground="green",
                                                                   command=collisions_button_clicked,
                                                                   font=(font, general_font_size), width=10)
                    CreateToolTip(collisions_frame.collisions_button,
                                  text='If this is turned on objects will merge when their radius`s intersect.')
                    collisions_label = tk.Label(collisions_frame, text="Collisions: ", font=(font, general_font_size))
                    collisions_label.pack(side=LEFT)
                    collisions_frame.collisions_button.pack(side=LEFT)
                    collisions_frame.pack(side=TOP, anchor=NW)

                def create_timestep_length(general_frame):
                    timestep_length_frame = tk.Frame(general_frame)
                    self.timestep_length_scale = LogScale(timestep_length_frame, from_=-5.0, to=0.0, orient=HORIZONTAL,
                                                     length=150, width=scale_width,
                                                     font=(font, general_font_size), resolution=1)
                    self.timestep_length_scale.set(-2)
                    timestep_length_label = tk.Label(timestep_length_frame, text="Timestep Length:",
                                                     font=(font, general_font_size))
                    timestep_length_label.pack(side=LEFT)
                    self.timestep_length_scale.pack(side=LEFT)
                    timestep_length_frame.pack(side=TOP, anchor=NW)

                general_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=100,
                                         highlightthickness=1)
                general_frame.grid(row=1, column=0, pady=0, padx=4)
                general_frame.pack_propagate(False)
                general = tk.Label(general_frame, text="General", font=(font, 15))
                general.pack(side=TOP)

                create_visuals_button(general_frame)
                create_collisions_button(general_frame)
                create_timestep_length(general_frame)

            # Creates the Widget for the multiple simulation controls
            def create_multiple_simulation_controls(control_frame):
                multiple_simulation_font_size = controls_font_size

                def create_number_simulations_slider(frame):
                    slider_frame = tk.Frame(frame)
                    self.number_simulations_scale = Scale(slider_frame, from_=1, to=100, orient=HORIZONTAL, length=150,
                                                          width=scale_width,
                                                          font=(font, multiple_simulation_font_size),
                                                          command=_screen_rules_updated)
                    self.number_simulations_scale.set(1)
                    slider_label = tk.Label(slider_frame, text="Number of Simulations:",
                                            font=font)
                    slider_label.pack(side=LEFT)
                    self.number_simulations_scale.pack(side=LEFT)
                    slider_frame.pack(side=TOP, anchor=NW)

                end_condition_specific_widgets = []

                def create_end_condition_dropdown(frame):
                    def end_condition_change(*args):
                        for w in end_condition_specific_widgets:
                            w.destroy()
                        if self.end_condition.get() == "Time or First Collision" or self.end_condition.get() == "Time":
                            create_end_time_slider(frame)

                    self.end_condition = StringVar()

                    options = ["Time", "First Collision", "Time or First Collision"]
                    # initial menu text
                    preset_frame = tk.Frame(frame)
                    drop = OptionMenu(preset_frame, self.end_condition, *options)
                    preset = tk.Label(preset_frame, text="End Simulation Condition: ",
                                      font=(font, multiple_simulation_font_size))
                    preset.pack(side=LEFT)
                    drop.pack(side=LEFT)
                    preset_frame.pack(side=TOP, anchor=NW)
                    self.end_condition.trace("w", end_condition_change)
                    self.end_condition.set("Time or First Collision")

                def create_end_time_slider(frame):
                    slider_frame = tk.Frame(frame)
                    end_condition_specific_widgets.append(slider_frame)
                    self.end_time_scale = Scale(slider_frame, from_=100, to=10000, orient=HORIZONTAL, length=150,
                                                width=scale_width,
                                                font=(font, multiple_simulation_font_size),
                                                command=_screen_rules_updated)
                    self.end_time_scale.set(1000)
                    slider_label = tk.Label(slider_frame, text="End Timestep:",
                                            font=font)
                    slider_label.pack(side=LEFT)
                    self.end_time_scale.pack(side=LEFT)
                    slider_frame.pack(side=TOP, anchor=NW)

                self.multiple_simulation_control_frame = tk.Frame(control_frame, highlightbackground="black", width=354,
                                                                  height=150,
                                                                  highlightthickness=1)
                self.multiple_simulation_control_frame.grid(row=2, column=0, pady=1, padx=4)
                self.multiple_simulation_control_frame.pack_propagate(False)
                label = tk.Label(self.multiple_simulation_control_frame, text="Multiple Simulations", font=(font, 15))
                label.pack(side=TOP)

                create_number_simulations_slider(self.multiple_simulation_control_frame)
                create_end_condition_dropdown(self.multiple_simulation_control_frame)

            # Creates the Widget for the visual controls
            def create_draw_controls(control_frame):
                draw_controls_font_size = controls_font_size

                def create_com_button(draw_control_frame):
                    self.center = "com"

                    def start_button_clicked():
                        if self.center == "com":
                            com_frame.com_button.config(highlightbackground="white", text="Tracking (0,0)")
                            self.center = "00"
                            _screen_rules_updated()
                        elif self.center == "00":
                            com_frame.com_button.config(highlightbackground="green", text="Tracking Largest Object")
                            self.center = "largest"
                            _screen_rules_updated()
                        elif self.center == "largest":
                            com_frame.com_button.config(highlightbackground="red", text="Tracking Center Of Mass")
                            self.center = "com"
                            _screen_rules_updated()

                    com_frame = tk.Frame(draw_control_frame)
                    com_frame.com_button = tk.Button(com_frame, text="Tracking Center Of Mass",
                                                     highlightbackground="green",
                                                     command=start_button_clicked, font=(font, draw_controls_font_size),
                                                     width=20)

                    com_label = tk.Label(com_frame, text="Center: ", font=(font, draw_controls_font_size))
                    com_label.pack(side=LEFT)
                    com_frame.com_button.pack(side=LEFT)
                    com_frame.pack(side=TOP, anchor=NW)

                def create_zoom(draw_control_frame):
                    zoom_frame = tk.Frame(draw_control_frame)
                    self.zoom_scale = Scale(zoom_frame, from_=1, to=100, orient=HORIZONTAL, length=150, width=scale_width,
                                       font=(font, draw_controls_font_size), showvalue=0, command=_screen_rules_updated)
                    self.zoom_scale.set(50)
                    zoom_label = tk.Label(zoom_frame, text="Zoom:",
                                          font=(font, draw_controls_font_size))
                    zoom_label.pack(side=LEFT)
                    self.zoom_scale.pack(side=LEFT)
                    zoom_frame.pack(side=TOP, anchor=NW)

                    def on_mousewheel(event):
                        if self.draw_visuals:
                            zoom = self.zoom_scale.get()
                            zoom += event.delta
                            self.zoom_scale.set(zoom)

                    self.canvas.bind_all("<MouseWheel>", on_mousewheel)

                def create_aparent_size(draw_control_frame):
                    aparent_size_frame = tk.Frame(draw_control_frame)
                    self.aparent_size_scale = Scale(aparent_size_frame, from_=1, to=150, orient=HORIZONTAL,
                                               length=150, width=scale_width, font=(font, draw_controls_font_size),
                                               showvalue=0, command=_screen_rules_updated)
                    self.aparent_size_scale.set(50)
                    aparent_size_label = tk.Label(aparent_size_frame, text="Aparent Size of Bodies:",
                                                  font=(font, draw_controls_font_size))
                    aparent_size_label.pack(side=LEFT)
                    self.aparent_size_scale.pack(side=LEFT)
                    aparent_size_frame.pack(side=TOP, anchor=NW)

                def create_path_button(draw_control_frame):
                    self.path_button_on = False

                    def start_button_clicked():
                        if self.path_button_on:
                            path_frame.path_button.config(highlightbackground="white", text="Not Showing Paths")
                            self.path_button_on = False
                        else:
                            path_frame.path_button.config(highlightbackground="green", text="Showing Paths")
                            self.path_button_on = True

                    path_frame = tk.Frame(draw_control_frame)
                    path_frame.path_button = tk.Button(path_frame, text="Not Showing Paths",
                                                       highlightbackground="white",
                                                       command=start_button_clicked,
                                                       font=(font, draw_controls_font_size),
                                                       width=20)

                    path_label = tk.Label(path_frame, text="Real Time Body Paths: ",
                                          font=(font, draw_controls_font_size))
                    path_label.pack(side=LEFT)
                    path_frame.path_button.pack(side=LEFT)
                    path_frame.pack(side=TOP, anchor=NW)

                def create_starting_path_button(draw_control_frame):
                    self.starting_path_button_on = False

                    def start_button_clicked():
                        _screen_rules_updated()
                        if self.starting_path_button_on:
                            starting_path_frame.starting_path_button.config(highlightbackground="white",
                                                                            text="Not Showing Starting Paths")
                            self.starting_path_button_on = False
                        else:
                            starting_path_frame.starting_path_button.config(highlightbackground="green",
                                                                            text="Showing Starting Paths")
                            self.starting_path_button_on = True

                    starting_path_frame = tk.Frame(draw_control_frame)
                    starting_path_frame.starting_path_button = tk.Button(starting_path_frame,
                                                                         text="Not Showing Starting Paths",
                                                                         highlightbackground="white",
                                                                         command=start_button_clicked,
                                                                         font=(font, draw_controls_font_size), width=25)

                    starting_path_label = tk.Label(starting_path_frame, text="Starting Body Paths: ",
                                                   font=(font, draw_controls_font_size))
                    starting_path_label.pack(side=LEFT)
                    starting_path_frame.starting_path_button.pack(side=LEFT)
                    starting_path_frame.pack(side=TOP, anchor=NW)

                draw_control_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=150,
                                              highlightthickness=1)
                draw_control_frame.grid(row=2, column=0, pady=1, padx=4)
                draw_control_frame.pack_propagate(False)
                draw_control_label = tk.Label(draw_control_frame, text="Visuals", font=(font, 15))
                draw_control_label.pack(side=TOP)

                create_com_button(draw_control_frame)
                create_zoom(draw_control_frame)
                create_aparent_size(draw_control_frame)
                create_path_button(draw_control_frame)
                create_starting_path_button(draw_control_frame)
                self.draw_control_frame = draw_control_frame

            # Creates the Widget for the starting controls
            def create_starting_controls(control_frame):
                starting_controls_font_size = controls_font_size

                def create_preset_menu(starting_frame):
                    options = [f for f in listdir("saved-start-conditions") if
                               isfile(join("saved-start-conditions", f))]
                    options.insert(0, "No Preset")
                    # initial menu text
                    start_condition.set("No Preset")
                    # Create Dropdown menu
                    preset_frame = tk.Frame(starting_frame)
                    drop = OptionMenu(preset_frame, start_condition, *options)
                    preset = tk.Label(preset_frame, text="Preset: ", font=(font, starting_controls_font_size))
                    preset.pack(side=LEFT)
                    drop.pack(side=LEFT)
                    preset_frame.pack(side=TOP, anchor=NW)

                def create_number_bodies(starting_frame):
                    number_bodies_frame = tk.Frame(starting_frame)
                    self.number_bodies_scale = NonLinearScale(number_bodies_frame, from_=2, to=150, orient=HORIZONTAL,
                                                         length=200,
                                                         width=scale_width, font=(font, starting_controls_font_size))
                    number_bodies_label = tk.Label(number_bodies_frame, text="Number of Bodies: ",
                                                   font=(font, starting_controls_font_size))
                    number_bodies_label.pack(side=LEFT)
                    self.number_bodies_scale.pack(side=LEFT)
                    number_bodies_frame.pack(side=TOP, anchor=NW)

                def create_distribution_area(starting_frame):
                    distribution_area_label = tk.Label(starting_frame, text="Distribution of Bodies:",
                                                       font=(font, starting_controls_font_size))
                    distribution_area_label.pack(side=TOP, anchor=NW)
                    distribution_area_frame = tk.Frame(starting_frame)
                    x_min = tk.StringVar()
                    x_max = tk.StringVar()
                    y_min = tk.StringVar()
                    y_max = tk.StringVar()
                    x_min.set("-200")
                    y_min.set("-200")
                    x_max.set("200")
                    y_max.set("200")
                    self.distribution_area = [x_min, x_max, y_min, y_max]
                    x_min_entry = tk.Entry(distribution_area_frame, textvariable=x_min, width=3,
                                           font=(font, starting_controls_font_size))
                    x_max_entry = tk.Entry(distribution_area_frame, textvariable=x_max, width=3,
                                           font=(font, starting_controls_font_size))
                    y_min_entry = tk.Entry(distribution_area_frame, textvariable=y_min, width=3,
                                           font=(font, starting_controls_font_size))
                    y_max_entry = tk.Entry(distribution_area_frame, textvariable=y_max, width=3,
                                           font=(font, starting_controls_font_size))
                    distribution_area_label1 = tk.Label(distribution_area_frame, text="Xmin:",
                                                        font=(font, starting_controls_font_size))
                    distribution_area_label2 = tk.Label(distribution_area_frame, text="Xmax:",
                                                        font=(font, starting_controls_font_size))
                    distribution_area_label3 = tk.Label(distribution_area_frame, text="Ymin:",
                                                        font=(font, starting_controls_font_size))
                    distribution_area_label4 = tk.Label(distribution_area_frame, text="Ymax:",
                                                        font=(font, starting_controls_font_size))

                    distribution_area_label1.pack(side=LEFT)
                    x_min_entry.pack(side=LEFT)
                    distribution_area_label2.pack(side=LEFT)
                    x_max_entry.pack(side=LEFT)
                    distribution_area_label3.pack(side=LEFT)
                    y_min_entry.pack(side=LEFT)
                    distribution_area_label4.pack(side=LEFT)
                    y_max_entry.pack(side=LEFT)
                    distribution_area_frame.pack(side=TOP, anchor=NW)

                def create_mass_entry(starting_frame):
                    mass_entry_frame = tk.Frame(starting_frame)
                    self.starting_mass = tk.StringVar()
                    self.starting_mass.set("100")
                    starting_mass_entry = tk.Entry(mass_entry_frame, textvariable=self.starting_mass, width=3,
                                                   font=(font, starting_controls_font_size))
                    starting_mass_label = tk.Label(mass_entry_frame, text="Starting Mass of Bodies:",
                                                   font=(font, starting_controls_font_size))
                    starting_mass_label.pack(side=LEFT)
                    starting_mass_entry.pack(side=LEFT)
                    mass_entry_frame.pack(side=TOP, anchor=NW)

                def create_radius_entry(starting_frame):
                    radius_entry_frame = tk.Frame(starting_frame)
                    self.starting_radius = tk.StringVar()
                    self.starting_radius.set("10")
                    starting_radius_entry = tk.Entry(radius_entry_frame, textvariable=self.starting_radius, width=3,
                                                     font=(font, starting_controls_font_size))
                    starting_radius_label = tk.Label(radius_entry_frame, text="Starting Radius of Bodies:",
                                                     font=(font, starting_controls_font_size))
                    starting_radius_label.pack(side=LEFT)
                    starting_radius_entry.pack(side=LEFT)
                    radius_entry_frame.pack(side=TOP, anchor=NW)

                def create_velocity_entry(starting_frame):
                    velocity_entry_frame = tk.Frame(starting_frame)
                    self.starting_velocity = tk.StringVar()
                    self.starting_velocity.set("1")
                    starting_velocity_entry = tk.Entry(velocity_entry_frame, textvariable=self.starting_velocity, width=3,
                                                       font=(font, starting_controls_font_size))
                    starting_velocity_label = tk.Label(velocity_entry_frame,
                                                       text="Average Starting Velocity of Bodies:",
                                                       font=(font, starting_controls_font_size))
                    starting_velocity_label.pack(side=LEFT)
                    starting_velocity_entry.pack(side=LEFT)
                    velocity_entry_frame.pack(side=TOP, anchor=NW)

                def create_rotation_entry(starting_frame):
                    rotation_entry_frame = tk.Frame(starting_frame)
                    self.starting_rotation = tk.StringVar()
                    self.starting_rotation.set("0")
                    starting_rotation_entry = tk.Entry(rotation_entry_frame, textvariable=self.starting_rotation, width=3,
                                                       font=(font, starting_controls_font_size))
                    starting_rotation_label = tk.Label(rotation_entry_frame,
                                                       text="Average Starting Rotation of Bodies:",
                                                       font=(font, starting_controls_font_size))
                    starting_rotation_label.pack(side=LEFT)
                    starting_rotation_entry.pack(side=LEFT)
                    rotation_entry_frame.pack(side=TOP, anchor=NW)

                starting_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=240,
                                          highlightthickness=1)
                starting_frame.grid(row=3, column=0, pady=0, padx=4)
                starting_frame.pack_propagate(False)
                starting = tk.Label(starting_frame, text="Starting Conditions", width=33, font=(font, 15))
                # starting.grid(row=0, column=0, columnspan=3, pady=2, padx=2)
                starting.pack(side=TOP)

                create_preset_menu(starting_frame)
                create_number_bodies(starting_frame)
                create_distribution_area(starting_frame)
                create_mass_entry(starting_frame)
                create_radius_entry(starting_frame)
                create_velocity_entry(starting_frame)
                create_rotation_entry(starting_frame)

            # Creates the Widget for the model controls
            def create_model_controls(control_frame):
                model_controls_font_size = controls_font_size
                model_specific_widgets = []

                def create_model_menu(model_frame):
                    global model_descriptions
                    global models
                    model_description = StringVar()

                    def selected_model_change(*args):
                        for w in model_specific_widgets:
                            w.destroy()
                        if self.selected_model.get() == "Dynamic Interval":
                            create_dynamic_interval_model_menu(model_frame)
                        model_description.set(model_descriptions[self.selected_model.get()])

                    self.selected_model = StringVar()
                    self.selected_model.trace("w", selected_model_change)
                    options = models
                    # initial menu text
                    self.selected_model.set(options[0])
                    # Create Dropdown menu
                    model_menu_frame = tk.Frame(model_frame)
                    model_menu_dropdown = OptionMenu(model_menu_frame, self.selected_model, *options)
                    CreateToolTip(model_menu_dropdown, text=model_description)
                    model_menu_label = tk.Label(model_menu_frame, text="Models: ",
                                                font=(font, model_controls_font_size))
                    model_menu_label.pack(side=LEFT)
                    model_menu_dropdown.pack(side=LEFT)
                    model_menu_frame.pack(side=TOP, anchor=NW)

                def create_dynamic_interval_model_menu(model_frame):
                    global dynamic_interval_scale
                    dynamic_interval_model_frame = tk.Frame(model_frame)
                    dynamic_interval_scale = Scale(dynamic_interval_model_frame, from_=1, to=100, orient=HORIZONTAL,
                                                   length=150)
                    CreateToolTip(dynamic_interval_scale,
                                  text='Modifier determing how much timesteps are slowed while objects are passing close, higher numbers lead to slower interactions',
                                  add_y=20)

                    dynamic_interval_model_label = tk.Label(dynamic_interval_model_frame, text="Interval Modifier: ",
                                                            font=(font, model_controls_font_size))
                    dynamic_interval_model_label.pack(side=LEFT)
                    dynamic_interval_scale.pack(side=LEFT)
                    dynamic_interval_model_frame.pack(side=TOP, anchor=NW)
                    model_specific_widgets.append(dynamic_interval_model_label)
                    model_specific_widgets.append(dynamic_interval_scale)
                    model_specific_widgets.append(dynamic_interval_model_frame)

                model_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=90,
                                       highlightthickness=1)
                model_frame.grid(row=4, column=0, pady=1, padx=4)
                model_frame.pack_propagate(False)
                model = tk.Label(model_frame, text="Simulation Model", font=(font, 15))
                model.pack(side=TOP)

                create_model_menu(model_frame)

            # Creates the Widget for the data controls
            def create_data_controls(control_frame):
                data_controls_font_size = controls_font_size

                def create_energy_graph_controls(data_frame):
                    energy_graph_top_frame = tk.Frame(data_frame)
                    self.graph_energy = False

                    def graph_energy_button_clicked():
                        if self.graph_energy:
                            energy_graph_top_frame.graph_energy_button.config(highlightbackground="white",
                                                                              text="Off")
                            self.graph_energy = False

                        else:
                            energy_graph_top_frame.graph_energy_button.config(highlightbackground="green",
                                                                              text="On")
                            self.graph_energy = True

                    energy_graph_top_frame.graph_energy_button = tk.Button(energy_graph_top_frame, text="Off",
                                                                           highlightbackground="white",
                                                                           command=graph_energy_button_clicked,
                                                                           font=(font, data_controls_font_size),
                                                                           width=20)

                    energy_graph_label = tk.Label(energy_graph_top_frame, text="Graph Energy: ",
                                                  font=(font, data_controls_font_size))
                    energy_graph_label.pack(side=LEFT)
                    energy_graph_top_frame.graph_energy_button.pack(side=LEFT)
                    energy_graph_top_frame.pack(side=TOP, anchor=NW)


                def create_actual_variance_graph_controls(data_frame):
                    actual_variance_graph_top_frame = tk.Frame(data_frame)
                    self.graph_actual_variance = False

                    def graph_actual_variance_button_clicked():
                        if self.graph_actual_variance:
                            actual_variance_graph_top_frame.graph_actual_variance_button.config(
                                highlightbackground="white",
                                text="Off")
                            self.graph_actual_variance = False

                        else:
                            actual_variance_graph_top_frame.graph_actual_variance_button.config(
                                highlightbackground="green",
                                text="On")

                            self.graph_actual_variance = True

                    actual_variance_graph_top_frame.graph_actual_variance_button = tk.Button(
                        actual_variance_graph_top_frame, text="Off",
                        highlightbackground="white",
                        command=graph_actual_variance_button_clicked,
                        font=(font, data_controls_font_size), width=20)

                    actual_variance_graph_label = tk.Label(actual_variance_graph_top_frame,
                                                           text="Graph Actual Variance: ",
                                                           font=(font, data_controls_font_size))
                    actual_variance_graph_label.pack(side=LEFT)
                    actual_variance_graph_top_frame.graph_actual_variance_button.pack(side=LEFT)
                    actual_variance_graph_top_frame.pack(side=TOP, anchor=NW)



                def create_graph_update_controls(data_frame):
                    graph_update_frame = tk.Frame(data_frame, bg="white")
                    self.graph_update_scale = Scale(graph_update_frame, from_=1, to=200,
                                                    orient=HORIZONTAL,
                                                    length=150, width=scale_width, bg="white",
                                                    font=(font, data_controls_font_size), showvalue=0)
                    self.graph_update_scale.set(100)
                    graph_update_label = tk.Label(graph_update_frame,
                                                                  text="Graph Update Interval:",
                                                                  font=(font, data_controls_font_size), bg="white")
                    graph_update_label.pack(side=LEFT)
                    self.graph_update_scale.pack(side=LEFT)
                    graph_update_frame.pack(side=TOP, anchor=NW, fill=X)

                def create_graph_x_axis_controls(data_frame):
                    graph_x_axis_frame = tk.Frame(data_frame, bg="white")
                    def graph_x_axis_button_clicked():
                        if not self.simulations_running:
                            if self.graph_x_axis == "actual_time":
                                graph_x_axis_frame.graph_x_axis_button.config(
                                    highlightbackground="white",
                                    text="Timestep")
                                self.graph_x_axis = "timestep"

                            elif self.graph_x_axis == "timestep":
                                graph_x_axis_frame.graph_x_axis_button.config(
                                    highlightbackground="green",
                                    text="Runtime")

                                self.graph_x_axis = "actual_time"

                    self.graph_x_axis = "timestep"
                    graph_x_axis_frame.graph_x_axis_button = tk.Button(
                        graph_x_axis_frame, text="Timestep",
                        highlightbackground="white",
                        command=graph_x_axis_button_clicked,
                        font=(font, data_controls_font_size), width=20)

                    actual_variance_graph_label = tk.Label(graph_x_axis_frame,
                                                           text="X Axis Variable for Graphs: ",
                                                           font=(font, data_controls_font_size))
                    actual_variance_graph_label.pack(side=LEFT)
                    graph_x_axis_frame.graph_x_axis_button.pack(side=LEFT)
                    graph_x_axis_frame.pack(side=TOP, anchor=NW, fill=X)

                def create_download_field(data_frame):
                    download_field_frame = tk.Frame(data_frame)
                    CreateToolTip(download_field_frame,
                                  text="Download graph data from the last simulations (will not work while running)")
                    download_file_name = tk.StringVar()

                    def download_field_button_clicked():
                        if not self.simulations_running:
                            if download_file_name.get() != "":
                                try:
                                    file = open("saved-simulation-data/" + download_file_name.get() + ".txt", "x")
                                except FileExistsError:
                                    return
                                file.write(download_file_name.get())
                                i = 0
                                for simulation in self.simulations:
                                    file.write("Simulation " + str(i) + "\n")
                                    file.write("Runtime: "+str(simulation.current_time)+"\n")
                                    file.write("Kinetic Energy:" + str(simulation.k_energy_list) + "\n")
                                    file.write(
                                        "Gravitational Potential Energy:" + str(simulation.k_energy_list) + "\n")
                                    if simulation.two_planet_start == True:
                                        file.write("PLanet 1 Variance:" + str(simulation.variances_list[0]) + "\n")
                                        file.write("PLanet 2 Variance:" + str(simulation.variances_list[1]) + "\n")
                                    i += 1

                    download_field_frame.download_fields_button = tk.Button(download_field_frame, text="Download",
                                                                            highlightbackground="white",
                                                                            command=download_field_button_clicked,
                                                                            font=(
                                                                                font,
                                                                                math.floor(data_controls_font_size)),
                                                                            width=9, activebackground="gray")

                    download_field_entry = tk.Entry(download_field_frame, textvariable=download_file_name, width=25,
                                                    font=(font, data_controls_font_size))

                    download_field_label = tk.Label(download_field_frame, text="FileName: ",
                                                    font=(font, data_controls_font_size))
                    download_field_label.pack(side=LEFT)
                    download_field_entry.pack(side=LEFT)
                    download_field_frame.download_fields_button.pack(side=LEFT)
                    download_field_frame.pack(side=TOP, anchor=NW)

                data_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=140,
                                      highlightthickness=1)
                data_frame.grid(row=5, column=0, pady=0, padx=4)
                data_frame.pack_propagate(False)
                data = tk.Label(data_frame, text="Data", font=(font, 15))
                data.pack(side=TOP)

                create_energy_graph_controls(data_frame)
                create_actual_variance_graph_controls(data_frame)
                create_graph_update_controls(data_frame)
                create_graph_x_axis_controls(data_frame)
                create_download_field(data_frame)


            # Creates the Widget for the starting button
            def create_start_button(control_frame):

                def start_button_clicked():
                    if not self.simulations_running:
                        control_frame.start_button.config(highlightbackground="red", text="End Simulation")
                        self.start_simulations(start_condition.get())
                    else:
                        control_frame.start_button.config(highlightbackground="green", text="Start Simulation")
                        self.end_simulations()

                control_frame.start_button = tk.Button(control_frame, text="Start Simulation",
                                                       highlightbackground="green",
                                                       command=start_button_clicked, width=16, font=(font, 20))
                control_frame.start_button.grid(row=6, column=0, pady=4, padx=4)

            # Creates the Controls Title
            control_frame = tk.Frame(self.window, highlightbackground="black", width=370, height=800,
                                     highlightthickness=2)
            control_frame.grid(row=1, column=0, rowspan=2, pady=8, padx=8)
            control_frame.grid_propagate(False)
            title = tk.Label(control_frame, text="Controls", font=(font, 25))
            title.grid(row=0, column=0, pady=0, padx=4)

            create_general_controls(control_frame)
            create_draw_controls(control_frame)
            create_starting_controls(control_frame)
            create_data_controls(control_frame)
            create_model_controls(control_frame)
            create_start_button(control_frame)

        # Creates the widget that contains the visuals
        def create_canvas():
            canvas_frame = tk.Frame(self.window, highlightbackground="black", width=800, height=800,
                                    highlightthickness=2)
            canvas_frame.grid(row=1, column=1, rowspan=2, pady=8, padx=8)
            self.canvas = tk.Canvas(canvas_frame, width=800, height=800)
            self.canvas.grid(row=0, column=0)
            self.drawn_objects = []

        # Creates the widget that contains the graphs
        def create_graphs():
            graphs_font_size = 20
            self.energy_graph_frame = tk.Frame(self.window, highlightbackground="black", width=370, height=390,
                                          highlightthickness=2)
            self.energy_graph_frame.grid(row=1, column=2, pady=8, padx=8)
            self.energy_graph_frame.grid_propagate(False)
            energy_graph_label = tk.Label(self.energy_graph_frame, text="Energy at each Timestep",
                                          font=(font, 25))
            energy_graph_label.grid(row=0, column=0)
            self.actual_variance_graph_frame = tk.Frame(self.window, highlightbackground="black", width=370, height=390,
                                                   highlightthickness=2)
            self.actual_variance_graph_frame.grid(row=2, column=2, pady=8, padx=8)
            self.actual_variance_graph_frame.grid_propagate(False)
            actual_variance_graph_label = tk.Label(self.actual_variance_graph_frame, text="Model and Actual Difference",
                                                   font=(font, 25))
            actual_variance_graph_label.grid(row=0, column=0)

        create_title()
        create_canvas()
        create_controls()
        create_graphs()

    # Handles the ending of all currently running simulations
    def end_simulations(self):
        for simulation in self.simulations:
            simulation.running = False
        print("simulations ended")

        self.simulations_running = False
        print("simulations ended")

    # Uses the user inputs to start on or multiple simulations
    def start_simulations(self, start_condition):
        self.simulations_running = True
        print("start")
        self.simulations = []
        if self.draw_visuals:
            self.start_simulation(start_condition)
        else:
            self.window.update()
            number_simulations = int(self.number_simulations_scale.get())
            for i in range(number_simulations):
                self.start_simulation(start_condition)
        self.simulations_running = False
        print("done")

    # Starts a single simulation
    def start_simulation(self, start_condition):
        planets = []
        self.canvas.delete("all")
        global planet_number
        planet_number = 0
        self.static_drawn_objects = []
        self.screen_rules_updated = True
        if start_condition == "No Preset":
            planets = []
            for i in range(0, int(self.number_bodies_scale.value)):
                pos = np.array([random.uniform(float(self.distribution_area[0].get()), float(self.distribution_area[1].get())),
                                random.uniform(float(self.distribution_area[2].get()), float(self.distribution_area[3].get()))])
                _starting_velocity = float(self.starting_velocity.get())
                distance = dist([0, 0], pos)
                rot = [-1 * random.uniform(0,float(self.starting_rotation.get())) * pos[1] / distance,
                       random.uniform(0,float(self.starting_rotation.get())) * pos[0] / distance]
                angle = 2*math.pi*random.random()
                speed = random.uniform(0, _starting_velocity)
                vel = np.array([rot[0] + np.cos(angle)*speed,
                                rot[1] + np.sin(angle)*speed])
                radius = float(self.starting_radius.get())
                mass = float(self.starting_mass.get())
                p = Planet(pos, vel, mass, radius)
                planets.append(p)
        else:
            file = open("saved-start-conditions/" + start_condition)
            lines = [line.rstrip() for line in file]
            for i in range(4, len(lines)):
                planet_info = lines[i].split(' ')
                pos = np.array(json.loads(planet_info[0][4:])).astype(np.float64)
                vel = np.array(json.loads(planet_info[1][4:])).astype(np.float64)
                radius = float(planet_info[2][7:])
                mass = float(planet_info[3][5:])
                path_color = planet_info[4][12:]
                color = planet_info[5][6:]
                p = Planet(pos, vel, mass, radius, path_color=path_color, color=color)
                planets.append(p)
        simulation = SimulationInstance(planets, self.selected_model.get(), self)
        self.simulations.append(simulation)
        simulation.run_simulation()

    # draws all the visuals for the passed simulation
    def draw(self, simulation):
        global window
        def get_scale( zoom_scale ):
            return (10 ** (float(zoom_scale.get()) / 25)) / 100

        scale = get_scale(self.zoom_scale)
        aparent_size = get_scale(self.aparent_size_scale)
        planets = simulation.planets
        com = simulation.com
        offset = [0,0]
        if self.center == "com":
            offset = com
        elif self.center == "largest":
            max_mass = 0
            for p in self.simulations[0].planets:
                if p.mass > max_mass:
                    max_mass = p.mass
                    offset = p.pos



        # draws one planet
        def drawPlanet(planet, com):

            x_offset = offset[0]
            y_offset = offset[1]

            xpos = planet.pos[0] - x_offset
            ypos = planet.pos[1] - y_offset
            self.drawn_objects.append(self.canvas.create_arc(scale * (xpos - aparent_size * planet.radius) + 400,
                                                   scale * (ypos - aparent_size * planet.radius) + 400,
                                                   scale * (xpos + aparent_size * planet.radius) + 400,
                                                   scale * (ypos + aparent_size * planet.radius) + 400,
                                                   start=0, extent=359.9, fill=planet.color, outline=""))

        # draws the center of mass
        def draw_com(com):
            x = 0
            _com = [ 400 + scale*(com[0] - offset[0]), 400 + scale*(com[1] - scale*offset[1])]
            self.drawn_objects.append(self.canvas.create_arc(_com[0] - 2, _com[1] - 2,
                                                   _com[0] + 2, _com[1] + 2,
                                                   start=0, extent=359, outline="", fill="red"))
            self.drawn_objects.append(self.canvas.create_line(_com[0] - 6, _com[1], _com[0] + 6, _com[1], fill="red"))
            self.drawn_objects.append(self.canvas.create_line(_com[0], _com[1] - 6, _com[0], _com[1] + 6, fill="red"))

        # draws a single path
        def draw_path(points, com, storage_array, color="black"):
            for i in range(0, len(points) - 2, 2):
                storage_array.append(self.canvas.create_line(400 + scale * (com[0] - offset[0] + points[i]),
                                                        400 + scale * (com[1] - offset[1] + points[i + 1]),
                                                        400 + scale * (com[0] - offset[0] + points[i + 2]),
                                                        400 + scale * (com[1] - offset[1] + points[i + 3]), width=1, fill=color))

        for o in self.drawn_objects:
            self.canvas.delete(o)
        self.drawn_objects = []

        for planet in planets:
            drawPlanet(planet, com)
        draw_com(com)
        if self.path_button_on and len(planets) == 2:
            paths = simulation.calculate_actual_paths()
            draw_path(paths[0], com, self.drawn_objects)
            draw_path(paths[1], com, self.drawn_objects)
        if (self.screen_rules_updated or self.center != "com") and self.starting_path_button_on and len(planets) <= 2 and simulation.starting_paths != None:

            starting_paths = simulation.starting_paths
            for o in self.static_drawn_objects:
                self.canvas.delete(o)
            draw_path(starting_paths[0], com, self.static_drawn_objects, color = "red")
            draw_path(starting_paths[1], com, self.static_drawn_objects, color = "red")
            self.screen_rules_updated = False

        self.canvas.update()

    # shows the graph of energy on the user interface
    def plot_energy(self, plotted_times, k_energys, gp_energys):
        x = plotted_times
        fig = Figure(figsize=(3.5, 3.5),
                     dpi=100)
        plot1 = fig.add_subplot(111)

        plot1.plot(x, k_energys, label="Kinetic Energy")
        plot1.plot(x, gp_energys, label="Gravitational Potential Energy")
        energys = []
        for i in range(len(k_energys)):
            energys.append(k_energys[i] + gp_energys[i])
        plot1.plot(x, energys, label="Total Energy")
        plot1.legend()
        # plot1.title.set_text("Energy vs Timestep")
        plot1.set_xlabel("timestep")
        plot1.set_ylabel("relative energy")
        old_graphs = self.energy_graph_frame.winfo_children()
        fig.subplots_adjust(top=1.0,
                            bottom=0.28,
                            left=0.25,
                            right=1.00,
                            hspace=0.01,
                            wspace=0.01)
        energy_graph_canvas = FigureCanvasTkAgg(fig,
                                                master=self.energy_graph_frame)
        energy_graph_canvas.draw()
        energy_graph_canvas.get_tk_widget().grid(row=1, column=0)
        for g in range(1, len(old_graphs)):
            old_graphs[g].destroy()

    # shows the graph of variance on the user interface
    def plot_variance(self, plotted_times, variances):
        x = plotted_times
        fig = Figure(figsize=(3.5, 3.5),
                     dpi=100)
        plot1 = fig.add_subplot(111)

        plot1.plot(x, variances[0], label="Planet 1 Variance")
        plot1.plot(x, variances[1], label="Planet 2 Variance")
        plot1.legend()
        plot1.set_xlabel("timestep")
        plot1.set_ylabel("distance from actual path")
        old_graphs = self.actual_variance_graph_frame.winfo_children()
        fig.subplots_adjust(top=1.0,
                            bottom=0.28,
                            left=0.2,
                            right=1.00,
                            hspace=0.01,
                            wspace=0.01)
        actual_variance_graph_canvas = FigureCanvasTkAgg(fig,
                                                         master=self.actual_variance_graph_frame)
        actual_variance_graph_canvas.draw()
        actual_variance_graph_canvas.get_tk_widget().grid(row=1, column=0)
        for g in range(1, len(old_graphs)):
            old_graphs[g].destroy()


if __name__ == '__main__':
    global window
    window = tk.Tk()
    user_interface = UserInterface(window)

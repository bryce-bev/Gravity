import tkinter as tk
from tkinter import *
import math
import time
import random
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from os import listdir
from os.path import isfile, join
import numpy as np
from queue import PriorityQueue

# G = 6.6742 * pow(10, -11)
G=1
font  = "Times"
canvas = ""
running = False
models = ["Interval", "Dynamic Interval"]
def dist(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2  +  (pos1[1]-pos2[1])**2)
def distSquared(pos1, pos2):
    return (pos1[0]-pos2[0])**2  +  (pos1[1]-pos2[1])**2
def componentX(pos1, pos2):
    return (pos2[0]-pos1[0]) / dist(pos1, pos2)
def componentY(pos1, pos2):
    return (pos2[1]-pos1[1]) / dist(pos1, pos2)
def get_scale(zoom_scale):
    return (10 ** (float(zoom_scale.get()) / 25)) / 100
def poly_oval(x0,y0, x1,y1, steps=100, rotation=0):
    rotation = rotation * math.pi / 180.0
    a = (x1 - x0) / 2.0
    b = (y1 - y0) / 2.0
    xc = x0 + a
    yc = y0 + b

    point_list = []
    for i in range(steps):
        theta = (math.pi * 2) * (float(i) / steps)
        x1 = a * math.cos(theta)
        y1 = b * math.sin(theta)
        x = (x1 * math.cos(rotation)) + (y1 * math.sin(rotation))
        y = (y1 * math.cos(rotation)) - (x1 * math.sin(rotation))
        point_list.append(round(x + xc))
        point_list.append(round(y + yc))

    return point_list
class NonLinearScale(tk.Scale):
    def __init__(self,parent,**kwargs):
        tk.Scale.__init__(self,parent,**kwargs)
        self.var = tk.DoubleVar()
        self['showvalue'] = 0
        self['label'] = 0
        self['command'] = self.update
        self['variable'] = self.var

    def update(self,event):
        self.config(label=self.value)

    @property
    def value(self):
        if int(self.var.get())<20:
            return str(int(self.var.get()))
        elif int(self.var.get())<50:
            return str(5 * (int(self.var.get()) - 16))
        else:
            return str(10 * (int(self.var.get()) - 35))

# https://stackoverflow.com/questions/69175812/in-a-tkinter-slider-can-you-vary-the-range-from-being-linear-to-exponential-for
class LogScale(tk.Scale):
    def __init__(self,parent,**kwargs):
        tk.Scale.__init__(self,parent,**kwargs)
        self.var = tk.DoubleVar()
        #self.config[showvalue] = 0
        self['showvalue'] = 0
        self['label'] = 0
        self['command'] = self.update
        self['variable'] = self.var

    def update(self,event):
        self.config(label=self.value)

    @property
    def value(self):
        return str(10**float(self.var.get()))
    def get(self):
        return 10 ** float(self.var.get())
class Planet:

    pos = np.array([0.0,0.0])
    vel = np.array([0.0,0.0])
    mass = 1
    radius = 1
    def __init__(self, pos, vel, mass, radius, path_color = "green", color = "black"):
        global planet_number
        self.planet_number = planet_number
        planet_number += 1
        self.last_pos = np.array([0.0,0.0])
        self.last_pos = np.copy(pos)
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.radius = radius
        self.next_update = 0
        self.path_color = path_color
        self.color = color

    def __gt__(self,other):
        if isinstance(other, Planet):
            return self.next_update > other.next_update

    def __str__(self):
     return "Pos: "+str(self.pos)+"  Vel: "+str(self.vel)+"  Mass: "+str(self.mass)+\
         "  Radius: "+str(self.radius)+"  Next Update: "+str(self.next_update)
def draw(planets,com):
    global canvas
    global window
    global drawn_objects
    global static_drawn_objects
    global path_button_on
    global starting_path_button_on
    global starting_paths
    global zoom_scale
    global aparent_size_scale
    scale = get_scale(zoom_scale)
    aparent_size = get_scale(aparent_size_scale)


    def drawPlanet(planet, com):
        global canvas

        x_offset = 0
        y_offset = 0
        if com_button_on:
            x_offset = 1 * com[0]
            y_offset = 1 * com[1]

        xpos = planet.pos[0] - x_offset
        ypos = planet.pos[1] - y_offset
        drawn_objects.append(canvas.create_arc(scale * (xpos - aparent_size * planet.radius) + 400,
                                               scale * (ypos - aparent_size * planet.radius) + 400,
                                               scale * (xpos + aparent_size * planet.radius) + 400,
                                               scale * (ypos + aparent_size * planet.radius) + 400,
                                               start=0, extent=359, fill=planet.color, outline=""))

    def draw_com(com):
        x = 0
        _com = [400, 400]
        global drawn_objects
        if not com_button_on:
            _com[0] = (scale * com[0]) + 400
            _com[1] = (scale * com[1]) + 400
        drawn_objects.append(canvas.create_arc(_com[0] - 2, _com[1] - 2,
                                               _com[0] + 2, _com[1] + 2,
                                               start=0, extent=359, outline="", fill="red"))
        drawn_objects.append(canvas.create_line(_com[0] - 6, _com[1], _com[0] + 6, _com[1], fill="red"))
        drawn_objects.append(canvas.create_line(_com[0], _com[1] - 6, _com[0], _com[1] + 6, fill="red"))

    def draw_path(points,com,storage_array):
        offset = [0, 0]
        if not com_button_on:
            offset = com
        for i in range(0, len(points) - 2, 2):
            storage_array.append(canvas.create_line(400+scale*(offset[0]+points[i]),
                                                    400+scale*(offset[1]+points[i + 1]),
                                                    400+scale*(offset[0]+points[i + 2]),
                                                    400+scale*(offset[1]+points[i + 3]), width=1))


    for o in drawn_objects:
        canvas.delete(o)
    drawn_objects = []

    for planet in planets:
        drawPlanet(planet,com)
    draw_com(com)
    if path_button_on and len(planets)==2:
        paths = calculate_actual_paths(planets[0],planets[1],com)
        draw_path(paths[0],com,drawn_objects)
        draw_path(paths[1],com,drawn_objects)
    global screen_rules_updated
    if screen_rules_updated and starting_path_button_on and len(planets)<=2:
        for o in static_drawn_objects:
            canvas.delete(o)
        draw_path(starting_paths[0], com, static_drawn_objects)
        draw_path(starting_paths[1], com, static_drawn_objects)
        screen_rules_updated = False

    canvas.update()

def calculate_actual_paths(p1, p2,com):
    com_vel = (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
    r1 = dist(com, p1.pos)
    r2 = dist(com, p2.pos)
    tangent1 = np.array([(p1.pos - com)[1], -1 * (p1.pos - com)[0]])
    tangent2 = np.array([(p2.pos - com)[1], -1 * (p2.pos - com)[0]])
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
        going_out = np.dot(planet.pos - com, planet.vel - com_vel) < 0
        left_side = np.degrees(np.arctan2(planet.vel[1] - com_vel[1], planet.vel[0] - com_vel[0])
                               - np.arctan2(planet.pos[1] - com[1], planet.pos[0] - com[0])) % 360 < 180
        if (going_out and not left_side) or (not going_out and left_side):
            sin = -1
        theta0 = sin * np.arccos((1 / ecc) * ((L ** 2 / (k * planet.mass * r)) - 1)) \
                 + np.arctan2((planet.pos[1] - com[1]), ((planet.pos[0] - com[0])))
        points = []
        if ecc < 1:
            assymptope_angle = 181
        else:
            assymptope_angle = 179-math.floor(np.degrees(np.arccos(1/ecc)))
        for theta in range(-1*assymptope_angle, assymptope_angle):
            theta = np.deg2rad(theta)
            radius = (L ** 2 / (planet.mass * k)) / (1 + ecc * np.cos(theta))
            points.append(radius * np.cos(theta + theta0))
            points.append(radius * np.sin(theta + theta0))
        return points

    p1_path = calculate_planet_path(p1, r1)
    p2_path = calculate_planet_path(p2, r2)
    return [p1_path, p2_path]
def plot_energy(plotted_times, k_energys, gp_energys):
    x = plotted_times
    fig = Figure(figsize=(3.5, 3.5),
                 dpi=100)
    plot1 = fig.add_subplot(111)

    plot1.plot(x, k_energys,label="Kinetic Energy")
    plot1.plot(x, gp_energys, label="Gravitational Potential Energy")
    energys = []
    for i in range(len(k_energys)):
        energys.append(k_energys[i]+gp_energys[i])
    plot1.plot(x, energys, label="Total Energy")
    plot1.legend()
    global energy_graph_frame
    old_graphs = energy_graph_frame.winfo_children()
    energy_graph_canvas = FigureCanvasTkAgg(fig,
                               master=energy_graph_frame )
    energy_graph_canvas.draw()
    energy_graph_canvas.get_tk_widget().grid(row=1,column=0)
    for g in range(1,len(old_graphs)):
         old_graphs[g].destroy()
def plot_variance(plotted_times, variances):
    x = plotted_times
    fig = Figure(figsize=(3.5, 3.5),
                 dpi=100)
    plot1 = fig.add_subplot(111)

    plot1.plot(x, variances[0], label="Planet 1 Variance")
    plot1.plot(x, variances[1], label="Planet 2 Variance")
    plot1.legend()
    global actual_variance_graph_frame
    old_graphs = actual_variance_graph_frame.winfo_children()
    actual_variance_graph_canvas = FigureCanvasTkAgg(fig,
                                            master=actual_variance_graph_frame)
    actual_variance_graph_canvas.draw()
    actual_variance_graph_canvas.get_tk_widget().grid(row=1, column=0)
    for g in range(1, len(old_graphs)):
        old_graphs[g].destroy()
def collide(planets, p1, p2):
    p1.pos[0] = (p1.pos[0] * p1.mass + p2.pos[0] * p2.mass) / (p1.mass + p2.mass)
    p1.pos[1] = (p1.pos[1] * p1.mass + p2.pos[1] * p2.mass) / (p1.mass + p2.mass)
    p1.vel[0] = (p1.vel[0] * p1.mass + p2.vel[0] * p2.mass) / (p1.mass + p2.mass)
    p1.vel[1] = (p1.vel[1] * p1.mass + p2.vel[1] * p2.mass) / (p1.mass + p2.mass)
    p1.mass += p2.mass
    p1.radius = ((p1.radius**3)+(p2.radius**3))**(1/3)
    planets.remove(p2)
def variance_from_path(planet,path,com):
    def dist_line_point(x1,y1,x2,y2, point):
        x0 = point[0]
        y0 = point[1]
        # equation taken from wikipedia
        return abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2)
    min = -1
    min_index = -1
    for i in range(0,len(path),2):
        distance = dist(np.array([path[i],path[i+1]]), planet.pos-com)
        if min==-1 or distance < min:
            min = distance
            min_index = i
    next_index = min_index+2
    prev_index = min_index - 2
    if min_index == len(path)-2:
        next_index = 0
    if min_index == 0:
        prev_index = len(path)-2
    if dist(np.array([path[next_index],path[next_index+1]]), planet.pos-com) < dist(np.array([path[prev_index],path[prev_index+1]]), planet.pos-com):
        return dist_line_point(path[min_index],path[min_index+1], path[next_index],path[next_index+1], planet.pos-com)
    else:
        return dist_line_point(path[prev_index],path[prev_index+1],path[min_index],path[min_index+1],  planet.pos-com)
def gp_energy(planets):
    gp_energy = 0
    for i in range(1,len(planets)):
        for j in range(0,i):
            gp_energy += -1*planets[i].mass*planets[j].mass / dist(planets[i].pos, planets[j].pos)

    return gp_energy
def k_energy(planets):
    k_energy = 0
    for p in planets:
        k_energy += 0.5 * p.mass * distSquared(p.vel,[0,0])
    return k_energy
def updatePlanet(p1, planets):
    maxA = 0
    min_dis = -1
    acceleration = np.array([0.0,0.0])
    global collisions
    collision = False
    for p2 in planets:
        if p1 != p2:
            if min_dis == -1 or min_dis > dist(p1.pos,p2.pos):
                if collisions and dist(p1.pos,p2.pos) < p1.radius+p2.radius:
                    collide(planets, p1, p2)
                    collision = True
                else:
                    min_dis = dist(p1.pos,p2.pos)
            if not (collision == True):
                acceleration[0] += componentX(p1.pos, p2.pos) * p2.mass / distSquared(p1.pos, p2.pos)
                acceleration[1] += componentY(p1.pos, p2.pos) * p2.mass / distSquared(p1.pos, p2.pos)
                collision = False
    velocity = p1.vel+acceleration
    global selected_model
    global dynamic_interval_scale
    if selected_model.get()=="Dynamic Interval":
        dynamic_interval_modifier = float(dynamic_interval_scale.get())
        if  dist([0,0], velocity) * dynamic_interval_modifier > min_dis :
            next_timestep = (1 * min_dis) / (dynamic_interval_modifier * dist([0,0], velocity))
        else:
            next_timestep = 1
    else:
        next_timestep = 1

    p1.next_update += next_timestep
    p1.vel += next_timestep * acceleration
    p1.last_pos = p1.pos
    p1.pos += next_timestep * p1.vel
def update_com(planets):
    com = np.array([0.0,0.0])
    totalMass = 0
    for p in planets:
        com += p.pos * p.mass
        totalMass += p.mass
    com = com / totalMass
    return com
def manager(planets):

    current_time = 0
    plotted_times = [0]
    k_energys = [k_energy(planets)]
    gp_energys = [gp_energy(planets)]
    model_actual_differences = [[0],[0]]
    planet_queue = PriorityQueue()
    for p in planets:
        planet_queue.put(p)
    com = update_com(planets)
    global running
    global draw_visuals
    global graph_energy
    global graph_actual_variance
    global energy_graph_update_scale
    global actual_variance_graph_update_scale
    global starting_paths
    global timestep_length_scale
    while running:
        p = planet_queue.get()
        if p in planets:
            timestep = p.next_update - current_time
            current_time = p.next_update
            updatePlanet(p, planets)
            com = update_com(planets)
            planet_queue.put(p)
            if math.floor(current_time - timestep) != math.floor(current_time) and not math.floor(current_time - timestep) in plotted_times:
                plotted_times.append(math.floor(current_time - timestep))
                if graph_energy:
                    gp_energys.append(gp_energy(planets))
                    k_energys.append(k_energy(planets))
                if graph_actual_variance:
                    for i in range(0,len(planets)):
                        model_actual_differences[i].append(variance_from_path(planets[i],starting_paths[i],com))
                if graph_energy and math.floor(current_time) % int(energy_graph_update_scale.get()) == 0:
                    plot_energy(plotted_times, k_energys, gp_energys)
                if graph_actual_variance and math.floor(current_time) % int(
                        actual_variance_graph_update_scale.get()) == 0:
                    plot_variance(plotted_times, model_actual_differences)
                if draw_visuals:
                    draw(planets, com)
                window.update()
                time.sleep(timestep_length_scale.get())


def start(start_condition):
    planets=[]
    global canvas
    canvas.delete("all")
    global planet_number
    planet_number = 0
    global static_drawn_objects
    static_drawn_objects = []
    global screen_rules_updated
    screen_rules_updated = True
    if start_condition == "No Preset":
        planets = []
        global number_bodies_scale
        global distribution_area
        global starting_mass
        global starting_radius
        global starting_velocity
        global starting_rotation
        for i in range(0,int(number_bodies_scale.value)):
            pos = np.array([random.uniform(float(distribution_area[0].get()),float(distribution_area[1].get())),
                   random.uniform(float(distribution_area[2].get()),float(distribution_area[3].get()))])
            _starting_velocity = float(starting_velocity.get())
            distance = dist([0,0],pos)
            rot = [-1*float(starting_rotation.get())*pos[1]/distance, float(starting_rotation.get())*pos[0]/distance]
            vel = np.array([rot[0]+random.uniform(-1*_starting_velocity,_starting_velocity), rot[1]+random.uniform(-1*_starting_velocity,_starting_velocity)])
            radius = float(starting_radius.get())
            mass = float(starting_mass.get())
            p = Planet(pos, vel, mass, radius)
            planets.append(p)
    else:
        file = open("saved-start-conditions/"+start_condition)
        lines = [line.rstrip() for line in file]
        for i in range(4,len(lines)):
            planet_info = lines[i].split(' ')
            pos = np.array(json.loads(planet_info[0][4:])).astype(np.float64)
            vel = np.array(json.loads(planet_info[1][4:])).astype(np.float64)
            radius = float(planet_info[2][7:])
            mass = float(planet_info[3][5:])
            path_color = planet_info[4][12:]
            color = planet_info[5][6:]
            p = Planet(pos, vel, mass, radius, path_color=path_color, color= color)
            planets.append(p)
    global starting_paths
    if len(planets) == 2:
        starting_paths = calculate_actual_paths(planets[0],planets[1],update_com(planets))
    manager(planets)



def create_UI(window):
    def create_title(window):
        title_frame = tk.Frame(window, highlightbackground="black", width=1580, height=55, highlightthickness=2)
        title_frame.grid(row=0, column=0, columnspan=3, pady=8, padx=8, sticky=(N, S, E, W))
        title = tk.Label(title_frame, text="N-Body Model Simulator", font=(font, 35))
        title.place(relx=0.5, rely=0.5, anchor=CENTER)
        # window.columnconfigure(1, weight=1)
        # window.rowconfigure(1, weight=1)


    def create_controls(window):
        start_condition = StringVar()
        controls_font_size= 12
        scale_width = 10


            # draw()
        def _screen_rules_updated(*args):
            global screen_rules_updated
            screen_rules_updated = True

        def create_general_controls(control_frame):
            general_font_size = controls_font_size
            def create_collisions_button(general_frame):
                global collisions
                collisions = True
                collisions_frame = tk.Frame(general_frame)

                def collisions_button_clicked():
                    global collisions
                    global canvas
                    if collisions:
                        collisions_frame.collisions_button.config(highlightbackground="red", text="Off")
                        collisions = False
                    else:
                        collisions_frame.collisions_button.config(highlightbackground="green", text="On")
                        collisions = True


                collisions_frame.collisions_button = tk.Button(collisions_frame, text="On",
                                                       highlightbackground="green",
                                                       command=collisions_button_clicked, font=(font, general_font_size),width=10)

                collisions_label = tk.Label(collisions_frame, text="Collisions: ", font=(font, general_font_size))
                collisions_label.pack(side=LEFT)
                collisions_frame.collisions_button.pack(side=LEFT)
                collisions_frame.pack(side=TOP, anchor=NW)

            def create_timestep_length(general_frame):
                global timestep_length_scale
                timestep_length_frame = tk.Frame(general_frame)
                timestep_length_scale = LogScale(timestep_length_frame, from_=-5.0, to=0.0, orient=HORIZONTAL, length=150, width=scale_width,
                                   font = (font,general_font_size), resolution=1)
                timestep_length_scale.set(-2)
                timestep_length_label = tk.Label(timestep_length_frame, text="Timestep Length:",
                                               font=(font, general_font_size))
                timestep_length_label.pack(side=LEFT)
                timestep_length_scale.pack(side=LEFT)
                timestep_length_frame.pack(side=TOP, anchor=NW)

            general_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=80,
                                     highlightthickness=1)
            general_frame.grid(row=1, column=0, pady=0, padx=4)
            general_frame.pack_propagate(False)
            general = tk.Label(general_frame, text="General", font=(font, 15))
            general.pack(side=TOP)

            create_collisions_button(general_frame)
            create_timestep_length(general_frame)

        def create_draw_controls(control_frame):
            draw_controls_font_size = controls_font_size
            def create_visuals_button(draw_control_frame):
                global draw_visuals
                draw_visuals = True
                draw_visuals_frame = tk.Frame(draw_control_frame)

                def draw_visuals_button_clicked():
                    global draw_visuals
                    global canvas
                    if draw_visuals:
                        draw_visuals_frame.draw_visuals_button.config(highlightbackground="red", text="Not Showing Simulation")
                        draw_visuals = False
                        canvas.create_rectangle(-10,370,820,430,fill="grey")
                        canvas.create_text(400,400,text="Not Showing Simulation",fill="black",font = (font, 40))
                    else:
                        draw_visuals_frame.draw_visuals_button.config(highlightbackground="green", text="Showing Simulation")
                        draw_visuals = True
                        canvas.delete("all")


                draw_visuals_frame.draw_visuals_button = tk.Button(draw_visuals_frame, text="Showing Simulation",
                                                       highlightbackground="green",
                                                       command=draw_visuals_button_clicked, font=(font, draw_controls_font_size),width=20)

                draw_visuals_label = tk.Label(draw_visuals_frame, text="Visuals: ", font=(font, draw_controls_font_size))
                draw_visuals_label.pack(side=LEFT)
                draw_visuals_frame.draw_visuals_button.pack(side=LEFT)
                draw_visuals_frame.pack(side=TOP, anchor=NW)

            def create_com_button(draw_control_frame):
                global com_button_on
                com_button_on = True

                def start_button_clicked():
                    global com_button_on
                    _screen_rules_updated()
                    if com_button_on:
                        com_frame.com_button.config(highlightbackground="white", text="Tracking (0,0)")
                        com_button_on = False
                        screen_rules_updated()
                    else:
                        com_frame.com_button.config(highlightbackground="green", text="Tracking Center Of Mass")
                        com_button_on = True
                        screen_rules_updated()

                com_frame = tk.Frame(draw_control_frame)
                com_frame.com_button = tk.Button(com_frame, text="Tracking Center Of Mass",
                                                       highlightbackground="green",
                                                       command=start_button_clicked, font=(font, draw_controls_font_size),width=20)

                com_label = tk.Label(com_frame, text="Center: ", font=(font, draw_controls_font_size))
                com_label.pack(side=LEFT)
                com_frame.com_button.pack(side=LEFT)
                com_frame.pack(side=TOP, anchor=NW)

            def create_zoom(draw_control_frame):
                global zoom_scale
                zoom_frame = tk.Frame(draw_control_frame)
                zoom_scale = Scale(zoom_frame, from_=1, to=100, orient=HORIZONTAL, length=150, width=scale_width,
                                   font = (font,draw_controls_font_size), showvalue=0, command = _screen_rules_updated)
                zoom_scale.set(50)
                zoom_label = tk.Label(zoom_frame, text="Zoom:",
                                               font=(font, draw_controls_font_size))
                zoom_label.pack(side=LEFT)
                zoom_scale.pack(side=LEFT)
                zoom_frame.pack(side=TOP, anchor=NW)
                def on_mousewheel(event):
                    zoom = zoom_scale.get()
                    zoom += event.delta
                    zoom_scale.set(zoom)


                global canvas
                canvas.bind_all("<MouseWheel>", on_mousewheel)

            def create_aparent_size(draw_control_frame):
                global aparent_size_scale
                aparent_size_frame = tk.Frame(draw_control_frame)
                aparent_size_scale = Scale(aparent_size_frame, from_=1, to=150, orient=HORIZONTAL,
                                           length=150, width=scale_width, font = (font,draw_controls_font_size),
                                           showvalue=0, command=_screen_rules_updated)
                aparent_size_scale.set(50)
                aparent_size_label = tk.Label(aparent_size_frame, text="Aparent Size of Bodies:",
                                               font=(font, draw_controls_font_size))
                aparent_size_label.pack(side=LEFT)
                aparent_size_scale.pack(side=LEFT)
                aparent_size_frame.pack(side=TOP, anchor=NW)

            def create_path_button(draw_control_frame):
                global path_button_on
                path_button_on = False

                def start_button_clicked():
                    global path_button_on
                    if path_button_on:
                        path_frame.path_button.config(highlightbackground="white", text="Not Showing Paths")
                        path_button_on = False
                    else:
                        path_frame.path_button.config(highlightbackground="green", text="Showing Paths")
                        path_button_on = True

                path_frame = tk.Frame(draw_control_frame)
                path_frame.path_button = tk.Button(path_frame, text="Not Showing Paths",
                                                       highlightbackground="white",
                                                       command=start_button_clicked, font=(font, draw_controls_font_size),width=20)

                path_label = tk.Label(path_frame, text="Real Time Body Paths: ", font=(font, draw_controls_font_size))
                path_label.pack(side=LEFT)
                path_frame.path_button.pack(side=LEFT)
                path_frame.pack(side=TOP, anchor=NW)

            def create_starting_path_button(draw_control_frame):
                global starting_path_button_on
                starting_path_button_on = False

                def start_button_clicked():
                    global starting_path_button_on
                    _screen_rules_updated()
                    if starting_path_button_on:
                        starting_path_frame.starting_path_button.config(highlightbackground="white", text="Not Showing Starting Paths")
                        starting_path_button_on = False
                    else:
                        starting_path_frame.starting_path_button.config(highlightbackground="green", text="Showing Starting Paths")
                        starting_path_button_on = True

                starting_path_frame = tk.Frame(draw_control_frame)
                starting_path_frame.starting_path_button = tk.Button(starting_path_frame, text="Not Showing Starting Paths",
                                                       highlightbackground="white",
                                                       command=start_button_clicked, font=(font, draw_controls_font_size),width=25)

                starting_path_label = tk.Label(starting_path_frame, text="Starting Body Paths: ", font=(font, draw_controls_font_size))
                starting_path_label.pack(side=LEFT)
                starting_path_frame.starting_path_button.pack(side=LEFT)
                starting_path_frame.pack(side=TOP, anchor=NW)

            draw_control_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=170,
                                          highlightthickness=1)
            draw_control_frame.grid(row=2, column=0, pady=1, padx=4)
            draw_control_frame.pack_propagate(False)
            draw_control_label = tk.Label(draw_control_frame, text="Visuals", font=(font, 15))
            draw_control_label.pack(side=TOP)

            create_visuals_button(draw_control_frame)
            create_com_button(draw_control_frame)
            create_zoom(draw_control_frame)
            create_aparent_size(draw_control_frame)
            create_path_button(draw_control_frame)
            create_starting_path_button(draw_control_frame)

        def create_starting_controls(control_frame):
            starting_controls_font_size = controls_font_size
            def create_preset_menu(starting_frame):
                options = [f for f in listdir("saved-start-conditions") if isfile(join("saved-start-conditions", f))]
                options.insert(0,"No Preset")
                # initial menu text
                start_condition.set("No Preset")
                # Create Dropdown menu
                preset_frame = tk.Frame(starting_frame)
                drop = OptionMenu(preset_frame, start_condition, *options)
                preset = tk.Label(preset_frame, text="Preset: ", font=(font, starting_controls_font_size))
                preset.pack(side=LEFT)
                drop.pack(side=LEFT)
                preset_frame.pack(side=TOP,anchor=NW)

            def create_number_bodies(starting_frame):
                global number_bodies_scale
                number_bodies_frame = tk.Frame(starting_frame)
                number_bodies_scale = NonLinearScale(number_bodies_frame, from_=2, to=95, orient=HORIZONTAL, length = 150, width=scale_width, font=(font, starting_controls_font_size))
                number_bodies_label = tk.Label(number_bodies_frame, text="Number of Bodies: ", font=(font, starting_controls_font_size))
                number_bodies_label.pack(side=LEFT)
                number_bodies_scale.pack(side=LEFT)
                number_bodies_frame.pack(side=TOP, anchor=NW)

            def create_distribution_area(starting_frame):
                distribution_area_label = tk.Label(starting_frame, text="Distribution of Bodies:", font=(font, starting_controls_font_size))
                distribution_area_label.pack(side=TOP, anchor=NW)
                distribution_area_frame = tk.Frame(starting_frame)
                global distribution_area
                x_min = tk.StringVar()
                x_max = tk.StringVar()
                y_min = tk.StringVar()
                y_max = tk.StringVar()
                x_min.set("-200")
                y_min.set("-200")
                x_max.set("200")
                y_max.set("200")
                distribution_area = [x_min,x_max,y_min,y_max]
                x_min_entry = tk.Entry(distribution_area_frame, textvariable = x_min,width=3, font=(font, starting_controls_font_size))
                x_max_entry = tk.Entry(distribution_area_frame, textvariable=x_max, width=3, font=(font, starting_controls_font_size))
                y_min_entry = tk.Entry(distribution_area_frame, textvariable=y_min, width=3, font=(font, starting_controls_font_size))
                y_max_entry = tk.Entry(distribution_area_frame, textvariable=y_max, width=3, font=(font, starting_controls_font_size))
                distribution_area_label1 = tk.Label(distribution_area_frame, text="Xmin:", font=(font, starting_controls_font_size))
                distribution_area_label2 = tk.Label(distribution_area_frame, text="Xmax:", font=(font, starting_controls_font_size))
                distribution_area_label3 = tk.Label(distribution_area_frame, text="Ymin:", font=(font, starting_controls_font_size))
                distribution_area_label4 = tk.Label(distribution_area_frame, text="Ymax:", font=(font, starting_controls_font_size))

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
                global starting_mass
                starting_mass = tk.StringVar()
                starting_mass.set("100")
                starting_mass_entry = tk.Entry(mass_entry_frame, textvariable=starting_mass, width=3, font=(font, starting_controls_font_size))
                starting_mass_label = tk.Label(mass_entry_frame, text="Starting Mass of Bodies:", font=(font, starting_controls_font_size))
                starting_mass_label.pack(side=LEFT)
                starting_mass_entry.pack(side=LEFT)
                mass_entry_frame.pack(side=TOP, anchor=NW)

            def create_radius_entry(starting_frame):
                radius_entry_frame = tk.Frame(starting_frame)
                global starting_radius
                starting_radius = tk.StringVar()
                starting_radius.set("10")
                starting_radius_entry = tk.Entry(radius_entry_frame, textvariable=starting_radius, width=3, font=(font, starting_controls_font_size))
                starting_radius_label = tk.Label(radius_entry_frame, text="Starting Radius of Bodies:", font=(font, starting_controls_font_size))
                starting_radius_label.pack(side=LEFT)
                starting_radius_entry.pack(side=LEFT)
                radius_entry_frame.pack(side=TOP, anchor=NW)

            def create_velocity_entry(starting_frame):
                velocity_entry_frame = tk.Frame(starting_frame)
                global starting_velocity
                starting_velocity = tk.StringVar()
                starting_velocity.set("1")
                starting_velocity_entry = tk.Entry(velocity_entry_frame, textvariable=starting_velocity, width=3, font=(font, starting_controls_font_size))
                starting_velocity_label = tk.Label(velocity_entry_frame, text="Average Starting Velocity of Bodies:", font=(font, starting_controls_font_size))
                starting_velocity_label.pack(side=LEFT)
                starting_velocity_entry.pack(side=LEFT)
                velocity_entry_frame.pack(side=TOP, anchor=NW)

            def create_rotation_entry(starting_frame):
                rotation_entry_frame = tk.Frame(starting_frame)
                global starting_rotation
                starting_rotation = tk.StringVar()
                starting_rotation.set("0")
                starting_rotation_entry = tk.Entry(rotation_entry_frame, textvariable=starting_rotation, width=3, font=(font, starting_controls_font_size))
                starting_rotation_label = tk.Label(rotation_entry_frame, text="Average Starting Rotation of Bodies:", font=(font, starting_controls_font_size))
                starting_rotation_label.pack(side=LEFT)
                starting_rotation_entry.pack(side=LEFT)
                rotation_entry_frame.pack(side=TOP, anchor=NW)

            starting_frame = tk.Frame(control_frame, highlightbackground="black", width=354,height=240,
                                      highlightthickness=1)
            starting_frame.grid(row=3, column=0, pady=0, padx=4)
            starting_frame.pack_propagate(False)
            starting = tk.Label(starting_frame, text="Starting Conditions",width=33, font=(font, 15))
            # starting.grid(row=0, column=0, columnspan=3, pady=2, padx=2)
            starting.pack(side=TOP)

            create_preset_menu(starting_frame)
            create_number_bodies(starting_frame)
            create_distribution_area(starting_frame)
            create_mass_entry(starting_frame)
            create_radius_entry(starting_frame)
            create_velocity_entry(starting_frame)
            create_rotation_entry(starting_frame)

        def create_model_controls(control_frame):
            model_controls_font_size = controls_font_size
            model_specific_widgets = []
            def create_model_menu(model_frame):
                global selected_model
                def selected_model_change(*args):
                    for w in model_specific_widgets:
                        w.destroy()
                    if selected_model.get() == "Dynamic Interval":
                        create_dynamic_interval_model_menu(model_frame)

                global models

                selected_model = StringVar()
                selected_model.trace("w", selected_model_change)
                options = models
                # initial menu text
                selected_model.set(options[0])
                # Create Dropdown menu
                model_menu_frame = tk.Frame(model_frame)
                model_menu_dropdown = OptionMenu(model_menu_frame, selected_model, *options)
                model_menu_label = tk.Label(model_menu_frame, text="Models: ", font=(font, model_controls_font_size))
                model_menu_label.pack(side=LEFT)
                model_menu_dropdown.pack(side=LEFT)
                model_menu_frame.pack(side=TOP, anchor=NW)

            def create_dynamic_interval_model_menu(model_frame):
                global dynamic_interval_scale
                dynamic_interval_model_frame = tk.Frame(model_frame)
                dynamic_interval_scale = Scale(dynamic_interval_model_frame, from_=1, to=100, orient=HORIZONTAL, length=150)
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

        def create_data_controls(control_frame):
            data_controls_font_size = controls_font_size
            def create_energy_graph_controls(data_frame):
                energy_graph_top_frame = tk.Frame(data_frame)
                energy_graph_bottom_frame = tk.Frame(data_frame, bg="gray")
                global energy_graph_update_scale
                energy_graph_update_scale = Scale(energy_graph_bottom_frame, from_=1, to=100, orient=HORIZONTAL,
                                                  length=150, width=scale_width, bg="gray", font=(font, data_controls_font_size))
                energy_graph_update_scale.set(5)
                energy_graph_update_scale.configure(state=DISABLED)         
                energy_graph_update_label = tk.Label(energy_graph_bottom_frame, text="Graph Update Interval:",
                                                     font=(font, data_controls_font_size), bg="gray")


                global graph_energy
                graph_energy = False

                def graph_energy_button_clicked():
                    global graph_energy
                    if graph_energy:
                        energy_graph_top_frame.graph_energy_button.config(highlightbackground="white",
                                                                      text="Off")
                        graph_energy = False
                        energy_graph_update_label.configure(bg="gray")
                        energy_graph_update_scale.configure(bg="gray")
                        energy_graph_update_scale.configure(state=DISABLED)
                        energy_graph_bottom_frame.configure(bg="gray")

                    else:
                        energy_graph_top_frame.graph_energy_button.config(highlightbackground="green",
                                                                      text="On")

                        graph_energy = True
                        energy_graph_update_label.configure(bg="white")
                        energy_graph_update_scale.configure(bg="white")
                        energy_graph_update_scale.configure(state=NORMAL)
                        energy_graph_bottom_frame.configure(bg="white")

                energy_graph_top_frame.graph_energy_button = tk.Button(energy_graph_top_frame, text="Off",
                                                                   highlightbackground="white",
                                                                   command=graph_energy_button_clicked,
                                                                   font=(font, data_controls_font_size), width=20)

                energy_graph_label = tk.Label(energy_graph_top_frame, text="Graph Energy: ",
                                              font=(font, data_controls_font_size))
                energy_graph_label.pack(side=LEFT)
                energy_graph_top_frame.graph_energy_button.pack(side=LEFT)
                energy_graph_top_frame.pack(side=TOP, anchor=NW)


                energy_graph_update_label.pack(side=LEFT)
                energy_graph_update_scale.pack(side=LEFT)
                energy_graph_bottom_frame.pack(side=TOP, anchor=NW, fill=X)

            def create_actual_variance_graph_controls(data_frame):
                actual_variance_graph_top_frame = tk.Frame(data_frame)
                actual_variance_graph_bottom_frame = tk.Frame(data_frame, bg="gray")
                global actual_variance_graph_update_scale
                actual_variance_graph_update_scale = Scale(actual_variance_graph_bottom_frame, from_=1, to=100, orient=HORIZONTAL,
                                                  length=150, width=scale_width, bg="gray", font=(font, data_controls_font_size))
                actual_variance_graph_update_scale.set(5)
                actual_variance_graph_update_scale.configure(state=DISABLED)
                actual_variance_graph_update_label = tk.Label(actual_variance_graph_bottom_frame, text="Graph Update Interval:",
                                                     font=(font, data_controls_font_size), bg="gray")


                global graph_actual_variance
                graph_actual_variance = False

                def graph_actual_variance_button_clicked():
                    global graph_actual_variance
                    if graph_actual_variance:
                        actual_variance_graph_top_frame.graph_actual_variance_button.config(highlightbackground="white",
                                                                      text="Off")
                        graph_actual_variance = False
                        actual_variance_graph_update_label.configure(bg="gray")
                        actual_variance_graph_update_scale.configure(bg="gray")
                        actual_variance_graph_update_scale.configure(state=DISABLED)
                        actual_variance_graph_bottom_frame.configure(bg="gray")

                    else:
                        actual_variance_graph_top_frame.graph_actual_variance_button.config(highlightbackground="green",
                                                                      text="On")

                        graph_actual_variance = True
                        actual_variance_graph_update_label.configure(bg="white")
                        actual_variance_graph_update_scale.configure(bg="white")
                        actual_variance_graph_update_scale.configure(state=NORMAL)
                        actual_variance_graph_bottom_frame.configure(bg="white")

                actual_variance_graph_top_frame.graph_actual_variance_button = tk.Button(actual_variance_graph_top_frame, text="Off",
                                                                   highlightbackground="white",
                                                                   command=graph_actual_variance_button_clicked,
                                                                   font=(font, data_controls_font_size), width=20)

                actual_variance_graph_label = tk.Label(actual_variance_graph_top_frame, text="Graph Actual Variance: ",
                                              font=(font, data_controls_font_size))
                actual_variance_graph_label.pack(side=LEFT)
                actual_variance_graph_top_frame.graph_actual_variance_button.pack(side=LEFT)
                actual_variance_graph_top_frame.pack(side=TOP, anchor=NW)


                actual_variance_graph_update_label.pack(side=LEFT)
                actual_variance_graph_update_scale.pack(side=LEFT)
                actual_variance_graph_bottom_frame.pack(side=TOP, anchor=NW, fill=X)

            data_frame = tk.Frame(control_frame, highlightbackground="black", width=354, height=140,
                                  highlightthickness=1)
            data_frame.grid(row=5, column=0, pady=0, padx=4)
            data_frame.pack_propagate(False)
            data = tk.Label(data_frame, text="Data", font=(font, 15))
            data.pack(side=TOP)

            create_energy_graph_controls(data_frame)
            create_actual_variance_graph_controls(data_frame)

        def create_start_button(control_frame):
            global running
            running = False

            def start_button_clicked():
                global running
                if not running:
                    control_frame.start_button.config(highlightbackground="red", text="End Simulation")
                    running = True
                    start(start_condition.get())
                else:
                    control_frame.start_button.config(highlightbackground="green", text="Start Simulation")
                    running = False

            control_frame.start_button = tk.Button(control_frame, text="Start Simulation", highlightbackground="green",
                                                   command=start_button_clicked, width=16, font=(font, 20))
            control_frame.start_button.grid(row=6, column=0, pady=4, padx=4)

        control_frame = tk.Frame(window, highlightbackground="black", width=370, height=800, highlightthickness=2)
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

    def create_canvas(window):
        canvas_frame = tk.Frame(window, highlightbackground="black", width=800, height=800, highlightthickness=2)
        canvas_frame.grid(row=1, column=1, rowspan=2, pady=8, padx=8)
        global canvas
        canvas = tk.Canvas(canvas_frame, width=800, height=800)
        canvas.grid(row=0,column=0)
        global drawn_objects
        drawn_objects = []
        return canvas

    def create_graphs(window):
        global energy_graph_frame
        energy_graph_frame = tk.Frame(window, highlightbackground="black", width=370, height=390, highlightthickness=2)
        energy_graph_frame.grid(row=1, column=2, pady=8, padx=8)
        energy_graph_frame.grid_propagate(False)
        energy_graph_label = tk.Label(energy_graph_frame, text="Distribution of Energy", font=(font, 25))
        energy_graph_label.grid(row=0,column=0)
        global actual_variance_graph_frame
        actual_variance_graph_frame = tk.Frame(window, highlightbackground="black", width=370, height=390, highlightthickness=2)
        actual_variance_graph_frame.grid(row=2, column=2, pady=8, padx=8)
        actual_variance_graph_frame.grid_propagate(False)
        actual_variance_graph_label = tk.Label(actual_variance_graph_frame, text="Model and Actual Difference", font=(font, 25))
        actual_variance_graph_label.grid(row=0, column=0)

    window.geometry("1600x900")
    window.configure(background="lightblue")
    create_title(window)
    create_canvas(window)
    create_controls(window)

    create_graphs(window)
    window.mainloop()
    # global running
    # running = True
    # start()

if __name__ == '__main__':
    print("start")

    global window
    window = tk.Tk()
    create_UI(window)











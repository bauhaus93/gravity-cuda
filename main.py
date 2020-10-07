#!/bin/env python3

import logging
import math
import random
import time

import numpy as np
import pygame
from numba import cuda


def setup_logger():
    FORMAT = r"[%(asctime)-15s] %(levelname)s - %(message)s"
    DATE_FORMAT = r"%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.ERROR, format=FORMAT, datefmt=DATE_FORMAT)
    logging.getLogger(__name__).setLevel(logging.INFO)


log = logging.getLogger(__name__)


def to_screen(pos, boundary, screen_size):
    return np.array(
        (
            (pos[0] - boundary[0]) / (boundary[2] - boundary[0]) * screen_size[0],
            (pos[1] - boundary[1]) / (boundary[3] - boundary[1]) * screen_size[1],
        ),
        dtype=int,
    )


def to_universe(pos, boundary, screen_size):
    return np.array(
        [
            (pos[0] / screen_size[0]) * (boundary[2] - boundary[0]) + boundary[0],
            (pos[1] / screen_size[1]) * (boundary[3] - boundary[1]) + boundary[1],
        ],
        dtype=float,
    )


@cuda.jit
def update_mass(masses_input, masses_output, delta_time):
    GRAVITATIONAL_CONSTANT = 6.674e-11
    (dest_index, src_index) = cuda.grid(2)

    if dest_index != src_index:
        direction_x = masses_input[src_index, 1] - masses_input[dest_index, 1]
        direction_y = masses_input[src_index, 2] - masses_input[dest_index, 2]
        distance = (direction_x ** 2 + direction_y ** 2) ** 0.5

        force = (
            GRAVITATIONAL_CONSTANT
            * (masses_input[dest_index, 0] * masses_input[src_index, 0])
            / distance
        )

        force_x = (direction_x / distance) * force
        force_y = (direction_y / distance) * force

        acc_x = force_x / masses_input[dest_index, 0] * delta_time
        acc_y = force_y / masses_input[dest_index, 0] * delta_time

        cuda.atomic.add(masses_output, (dest_index, 3), acc_x)
        cuda.atomic.add(masses_output, (dest_index, 4), acc_y)
    cuda.syncthreads()
    if src_index == 0:
        masses_output[dest_index][1] += masses_output[dest_index][3] * delta_time
        masses_output[dest_index][2] += masses_output[dest_index][4] * delta_time


def update_masses(masses_input, delta_time):
    masses_output = masses_input.copy()

    input_device = cuda.to_device(masses_input)
    output_device = cuda.to_device(masses_output)

    mass_count = masses_input.shape[0]
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(mass_count / threads_per_block[0])
    blocks_per_grid_y = math.ceil(mass_count / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    update_mass[blocks_per_grid, threads_per_block](
        input_device, output_device, delta_time
    )
    output_device.to_host()
    return masses_output


def create_random_masses(mass_count, max_mass, boundary):
    mass_list = np.zeros((mass_count, 5))
    for i in range(mass_count):
        mass = random.uniform(1.0, max_mass)
        pos_x = random.uniform(boundary[0], boundary[2])
        pos_y = random.uniform(boundary[1], boundary[3])
        mass_list[
            i,
        ] = (mass, pos_x, pos_y, 0.0, 0.0)
    return mass_list


class Universe:
    def __init__(self, mass_count, max_mass, initial_size):
        self.initital_size = np.array(initial_size, dtype=float)
        self.boundary = np.array(
            (0.0, 0.0, initial_size[0], initial_size[1]), dtype=float
        )
        self.max_mass = max_mass

        self.mass_list = create_random_masses(
            mass_count, max_mass, np.array((0.0, 0.0, initial_size[0], initial_size[1]))
        )

    def add_mass(self, universe_pos, mass):
        self.mass_list = np.append(
            self.mass_list,
            [[mass, universe_pos[0], universe_pos[1], 0.0, 0.0]],
            axis=0,
        )
        if mass > self.max_mass:
            self.max_mass = mass

    def zoom(self, factor):
        new_size = self.get_size() * factor
        center = self.get_center()
        sign = [-1.0, -1.0, 1.0, 1.0]
        for i in range(len(self.boundary)):
            self.boundary[i] = center[i % 2] + sign[i] * new_size[i % 2] / 2.0

    def center(self, point):
        size = self.get_size()
        self.boundary = np.array(
            [
                point[0] - size[0] / 2,
                point[1] - size[1] / 2,
                point[0] + size[0] / 2,
                point[1] + size[1] / 2,
            ],
            dtype=float,
        )

    def get_center(self):
        size = self.get_size()
        return np.array(
            [self.boundary[0] + size[0] / 2.0, self.boundary[1] + size[1] / 2.0],
            dtype=float,
        )

    def get_boundary(self):
        return self.boundary

    def get_mass_count(self):
        return self.mass_list.shape[0]

    def tick(self, delta_time):
        self.mass_list = update_masses(self.mass_list, delta_time)

    def fit_boundary_to_masses(self, smooth=False):
        needs_expand = False
        for i in range(self.mass_list.shape[0]):
            if smooth:
                self.boundary[0] = min(self.boundary[0], self.mass_list[i, 1])
                self.boundary[1] = min(self.boundary[1], self.mass_list[i, 2])
                self.boundary[2] = max(self.boundary[2], self.mass_list[i, 1])
                self.boundary[3] = max(self.boundary[3], self.mass_list[i, 2])
            else:
                if self.boundary[0] > self.mass_list[i, 1]:
                    needs_expand = True
                elif self.boundary[1] > self.mass_list[i, 2]:
                    needs_expand = True
                elif self.boundary[2] < self.mass_list[i, 1]:
                    needs_expand = True
                elif self.boundary[3] < self.mass_list[i, 2]:
                    needs_expand = True
                if needs_expand:
                    break

        if needs_expand:
            expand = [-1.0, -1.0, 1.0, 1.0]
            size = self.get_size()
            for i in range(4):
                self.boundary[i] += expand[i] * size[i % 2] / 2.0

    def get_size(self):
        return np.array(
            (self.boundary[2] - self.boundary[0], self.boundary[3] - self.boundary[1])
        )

    def get_max_velocity(self):
        max_vel = 0
        for i in range(self.mass_list.shape[0]):
            vel = (self.mass_list[i][3] ** 2 + self.mass_list[i][4] ** 2) ** 0.5
            max_vel = max(max_vel, vel)
        return max_vel

    def get_scale_factor(self):
        return self.get_size() / self.initital_size

    def draw(self, surface):
        surface_size = surface.get_size()
        for i in range(self.mass_list.shape[0]):
            screen_pos = to_screen(self.mass_list[i][1:3], self.boundary, surface_size)
            if (
                screen_pos[0] >= 0
                and screen_pos[0] < surface_size[0]
                and screen_pos[1] >= 0
                and screen_pos[1] < surface_size[1]
            ):
                mass_frac = self.mass_list[i][0] / self.max_mass
                color = np.array(
                    (0xFF * mass_frac, 0xFF * (1.0 - mass_frac), 0), dtype=int
                )
                pygame.draw.circle(surface, color, screen_pos, 2)


def draw_scene(display, surface, font, universe, update_time):
    BACKGROUND_COLOR = (0, 0, 0)
    surface.fill(BACKGROUND_COLOR)

    universe.draw(surface)

    scale = universe.get_scale_factor()
    max_vel = universe.get_max_velocity()
    update_time_surface = font.render(
        f"Update time: {round(update_time):.0f} ms",
        True,
        [0xFF] * 3,
        [0] * 3,
    )
    mass_count_surface = font.render(
        f"Mass count: {universe.get_mass_count()}",
        True,
        [0xFF] * 3,
        [0] * 3,
    )
    scale_surface = font.render(
        f"Scale:      {scale[0]:.1f}/{scale[1]:.1f}",
        True,
        [0xFF] * 3,
        [0] * 3,
    )
    vel_surface = font.render(
        f"Max Vel:  {max_vel / 1e3:.1f} km/s",
        True,
        [0xFF] * 3,
        [0] * 3,
    )
    display.blit(surface, (0, 0))
    display.blit(update_time_surface, (0, 20))
    display.blit(mass_count_surface, (0, 40))
    display.blit(scale_surface, (0, 60))
    display.blit(vel_surface, (0, 80))
    pygame.display.flip()


if __name__ == "__main__":
    SCREEN_SIZE = (1024, 768)
    UNIVERSE_SIZE = (1e6, 1e6)
    MAX_MASS = 1e14
    MASS_COUNT = 1000
    DELTA_TIME = 1.0
    setup_logger()
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 30)
    display = pygame.display.set_mode(SCREEN_SIZE)
    surface = pygame.Surface(display.get_size())
    surface.convert()

    universe = Universe(MASS_COUNT, MAX_MASS, UNIVERSE_SIZE)

    quit = False
    delta_time = DELTA_TIME
    while not quit:
        start = time.time()
        universe.tick(delta_time)
        update_time = (time.time() - start) * 1000.0
        draw_scene(display, surface, font, universe, update_time)
        render_time = (time.time() - start) * 1000.0

        for e in pygame.event.get():
            if e.type == pygame.constants.QUIT:
                quit = True
            elif e.type == pygame.MOUSEBUTTONDOWN:
                ZOOM_FACTOR = 1.2
                if e.button == 1:
                    universe_pos = to_universe(
                        e.pos, universe.get_boundary(), SCREEN_SIZE
                    )
                    universe.center(universe_pos)
                elif e.button == 3:
                    universe_pos = to_universe(
                        e.pos, universe.get_boundary(), SCREEN_SIZE
                    )
                    universe.add_mass(universe_pos, MAX_MASS * 2.0)
                elif e.button == 4:  # wheel up
                    universe.zoom(1.0 / ZOOM_FACTOR)
                elif e.button == 5:
                    universe.zoom(ZOOM_FACTOR)
        # pygame.time.delay(20)

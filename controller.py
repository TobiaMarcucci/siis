import numpy as np
from pydrake.all import VectorSystem

class Controller(VectorSystem):

    def __init__(self, pusher_slider):

        # 17 inputs state of the pusher slider
        nx = pusher_slider.num_positions() + pusher_slider.num_velocities()

        # 2 outputs force on the pusher
        nu = pusher_slider.num_actuators()

        # instantiate controller
        VectorSystem.__init__(self, nx, nu)
        self.pusher_slider = pusher_slider

    def DoCalcVectorOutput(self, context, x, _, u):

        # extract positions from state
        pos_p = x[7:9]
        pos_s = np.concatenate([x[4:6],[2*np.arctan2(x[3], x[0])]])

        # extract velocities from state
        vel_p = x[15:17]
        vel_s = x[9:15]

        # check if slider orientation is in limits
        tol = 1.e-2
        target, error = self.alignment_error(pos_s[-1])

        # block pusher in current position
        if abs(error) < tol:
            u[:] = self.pusher_pd(pos_p, pos_p, vel_p)
            return

        # ensure that slider is still
        tol = 1.e0
        still = np.linalg.norm(vel_s) < tol

        # block pusher if not still
        if not still:
            u[:] = self.pusher_pd(pos_p, pos_p, vel_p)
            return

        # express pusher position in slider frame
        pos_ps, R = self.pusher_in_slider_frame(pos_p, pos_s)

        # detect contact
        contact, facet = self.detect_contact(pos_ps)

        # approach slider
        if not contact:
            pos_p_des = self.get_pushing_point(error, facet, R, pos_s)
            u[:] = self.pusher_pd(pos_p_des, pos_p, vel_p)

        # rotate slider
        else:
            u[:] = self.rotate_slider(error, pos_s[-1], vel_s[2], facet)

    def detect_contact(self, pos_ps):

        # find closest facet
        pos_ps_abs = np.concatenate([pos_ps, -pos_ps])
        facet = np.argmax(pos_ps_abs)

        # sistem sizes
        slider_half_side = .5
        pusher_radius = .1
        tol = 5.e-3
        l = slider_half_side + pusher_radius + tol

        # check contact
        contact = pos_ps_abs[facet] < l

        return contact, facet

    def pusher_in_slider_frame(self, pos_p, pos_s):
        t = pos_s[-1]
        R = np.array([
            [  np.cos(t), np.sin(t)],
            [- np.sin(t), np.cos(t)]
            ])
        return R.dot(pos_p - pos_s[:2]), R

    def alignment_error(self, t):

        # list of acceptable angles
        targets = np.pi * np.array([0, .5, 1, 1.5, 2.])

        # find closest
        t_wrapped = t % (2*np.pi)
        errors = t_wrapped - targets
        argmin = np.argmin(abs(errors))

        return targets[argmin], errors[argmin]

    def pusher_pd(self, pos_p_des, pos_p, vel_p):

        # control gains for stabilizing pusher
        kp = 10
        kd = 1

        return - kp * (pos_p-pos_p_des) - kd * vel_p

    def get_pushing_point(self, error, facet, R, pos_s):

        # find point to push depending on error sign
        if error > 0:
            pushing_points = [
            np.array([.6, -.4]),
            np.array([.4, .6]),
            np.array([-.6, .4]),
            np.array([-.4, -.6]),
            ]
        else:
            pushing_points = [
            np.array([.6, .4]),
            np.array([-.4, .6]),
            np.array([-.6, -.4]),
            np.array([.4, -.6]),
            ]

        # express pushing point in absolute frame
        pushing_point = np.linalg.inv(R).dot(pushing_points[facet]) + pos_s[:2]

        return pushing_point

    def rotate_slider(self, error, t, td, facet):

        # # control gains for stabilizing slider
        # kp = 100
        # kd = 30

        # # derive contact force
        # force_magnitude = kp * error + kd * td

        # just use constant force to overcome friction
        force_magnitude = 10

        facet_angle = t + facet * np.pi/2
        u = - force_magnitude * np.array([np.cos(facet_angle), np.sin(facet_angle)])

        return u
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, filename='humanoid_openai.xml'):
        mujoco_env.MujocoEnv.__init__(self, filename, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def extremities(self):
        torso_frame = self.data.get_body_xmat('thorax').reshape(3, 3)
        torso_pos = self.data.get_body_xpos('thorax')
        positions = []
        for side in ('l', 'r'):
            torso_to_limb = self.data.get_body_xpos(side + 'foot') - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        return np.concatenate(positions)

    def ego_shape(self):
        return np.squeeze(self.get_egocentric().shape)

    def ego_pure_shape(self):
        return np.squeeze(self.get_pure_egocentric().shape)
        
    def get_pure_egocentric(self):
        root_upright = [self.sim.data.get_body_xmat('thorax')[2, 1]]
        root_height = [self.sim.data.get_body_xpos('thorax')[2]]
        center_of_mass_position = self.sim.data.subtree_com[-1]
        center_of_mass_velocity = self.sim.data.subtree_linvel[-1]
        torso_vertical_orientation = self.sim.data.get_body_xmat('thorax')[2, :]
        sensordata = self.data.sensordata[0:9]
        extremities = self.extremities()
        data = np.concatenate([root_upright, root_height, center_of_mass_position,
                               center_of_mass_velocity, torso_vertical_orientation,
                               sensordata, extremities])
        return data

    def get_egocentric(self):
        qpos = self.sim.data.qpos.flat[2:]
        qvel = self.sim.data.qvel.flat
        root_upright = [self.sim.data.get_body_xmat('thorax')[2, 1]]
        root_height = [self.sim.data.get_body_xpos('thorax')[2]]
        center_of_mass_position = self.sim.data.subtree_com[-1]
        center_of_mass_velocity = self.sim.data.subtree_linvel[-1]
        torso_vertical_orientation = self.sim.data.get_body_xmat('thorax')[2, :]
        sensordata = self.data.sensordata[0:9]
        extremities = self.extremities()
        data = np.concatenate([qpos, qvel,
                               root_upright, root_height, center_of_mass_position,
                               center_of_mass_velocity, torso_vertical_orientation,
                               sensordata, extremities])
        return data

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 0.6) or (qpos[2] > 1.4))
        return self._get_obs(), reward, done, dict(egocentric_feats=self.get_egocentric(), pure_egocentric_feats=self.get_pure_egocentric(), reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)
        
    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

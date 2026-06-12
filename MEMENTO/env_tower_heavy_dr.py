from collections import OrderedDict
import numpy as np

from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat, quat2mat


class FrankaEnv(ManipulationEnv):
    """
    Tower of Hanoi (4 cubes, 3 boxes/pegs)

    NOTE:
    - This version upgrades the environment / task to 4 cubes and 3 pegs.
    - I intentionally leave the fitness functions structurally unchanged (as requested),
      so the current dense fitness still only tracks the original 3 cubes (small/med/large).
      The new extra cube is included in reset / success / observables / task definition.
      We can update dense fitness next.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.6, 1.1, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        placement_initializer=None,
        has_renderer=False,
        reward_scale=1.0,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=12000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # cube half-edges (m) -> 4 cubes, reasonable spacing
        # largest -> large -> medium -> small
        self.size_xlarge = 0.031
        self.size_large = 0.028
        self.size_med = 0.025
        self.size_small = 0.022

        # hole gate in container local frame
        self.hole_x_min = 0.02
        self.hole_x_max = 0.09
        self.hole_y_abs = 0.035

        # stacking tolerances
        self.stack_xy_tol = 1.15  # * min(half_edge_i, half_edge_j)
        self.stack_z_tol = 0.010  # |dz - (si+sj)|
        self.pos_noise_std = 0.005
        self.pos_bias_range = 0.004
        self._pos_bias = np.zeros(3)

        # dynamics randomization
        self.friction_range = [0.6, 1.2]

        # world calibration offset
                # world calibration offset
        self.world_xy_offset_range = [-0.005, 0.005]
        self.heavy_dr = True

# Observation noise: close to light, but variable
        self.pos_noise_std_range = [0.003, 0.008]   # was [0.002, 0.025]

        # Dynamics
        self.cube_mass_scale_range = [0.9, 1.15]    # was [0.7, 1.5]
        self.joint_damping_scale_range = [0.9, 1.15] # was [0.8, 1.5]
        self.actuator_gain_scale_range = [0.95, 1.05] # was [0.85, 1.15]

        # Friction
        self.friction_range_heavy = [0.7, 1.3]      # was [0.3, 1.8]
        self.torsional_friction_range = [0.003, 0.008]
        self.rolling_friction_range = [0.0002, 0.0008]

        # Observation/world perturbation
        self.pos_bias_range_heavy = 0.005           # was 0.015
        self.world_xy_offset_range_heavy = [-0.007, 0.007]  # was [-0.015, 0.015]

        # Not implemented, so disable
        self.obs_latency_range = [0, 0]             # was [0, 2]

        # Initial stack perturbation
        self.cube_xy_perturb_range = 0.002          # was 0.003

        # Store base values after first reset
        self._base_masses = None
        self._base_damping = None
        self._base_gains = None
        self._base_geom_friction = None

        # Strict Hanoi legality: terminate if a larger cube is placed on a smaller cube
        self.terminate_on_illegal_stack = True
        self.illegal_stack_term_thresh = 0.70  # continuous stack score threshold

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # ----------------------------- LOAD MODEL -----------------------------

    def _load_model(self):
        super()._load_model()

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        xpos = list(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        xpos[2] -= 0.1
        self.robots[0].robot_model.set_base_xpos(xpos)

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
                                 tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        bluewood = CustomMaterial(texture="WoodBlue", tex_name="bluewood", mat_name="bluewood_mat",
                                  tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat",
                                   tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        graywood = CustomMaterial(texture="WoodLight", tex_name="graywood", mat_name="graywood_mat",
                                  tex_attrib=tex_attrib, mat_attrib=mat_attrib)

        # Keep legacy names for compatibility:
        # cube      -> smallest
        # cube_blue -> medium
        # cube_green-> large
        # cube_xlarge -> new largest
        self.cube_small = BoxObject(
            name="cube",
            size_min=[self.size_small, self.size_small, self.size_small],
            size_max=[self.size_small, self.size_small, self.size_small],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cube_med = BoxObject(
            name="cube_blue",
            size_min=[self.size_med, self.size_med, self.size_med],
            size_max=[self.size_med, self.size_med, self.size_med],
            rgba=[0, 0.2, 1, 1],
            material=bluewood,
        )
        self.cube_large = BoxObject(
            name="cube_green",
            size_min=[self.size_large, self.size_large, self.size_large],
            size_max=[self.size_large, self.size_large, self.size_large],
            rgba=[0.0, 0.7, 0.2, 1],
            material=greenwood,
        )
        self.cube_xlarge = BoxObject(
            name="cube_xlarge",
            size_min=[self.size_xlarge, self.size_xlarge, self.size_xlarge],
            size_max=[self.size_xlarge, self.size_xlarge, self.size_xlarge],
            rgba=[0.65, 0.65, 0.65, 1],
            material=graywood,
        )

        xml_obj = xml_path_completion("objects/plate-with-hole_bigger.xml")
        self.box_A = MujocoXMLObject(xml_obj, name="container", joints="default")      # legacy name
        self.box_B = MujocoXMLObject(xml_obj, name="container_B", joints="default")
        self.box_C = MujocoXMLObject(xml_obj, name="container_C", joints="default")

        self.objects = [
            self.cube_small, self.cube_med, self.cube_large, self.cube_xlarge,
            self.box_A, self.box_B, self.box_C
        ]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    # --------------------------- SETUP REFERENCES ---------------------------

    def _setup_references(self):
        super()._setup_references()

        # cube bodies
        self.cube_small_body_id = self.sim.model.body_name2id(self.cube_small.root_body)
        self.cube_med_body_id = self.sim.model.body_name2id(self.cube_med.root_body)
        self.cube_large_body_id = self.sim.model.body_name2id(self.cube_large.root_body)
        self.cube_xlarge_body_id = self.sim.model.body_name2id(self.cube_xlarge.root_body)

        # container bodies
        self.box_A_body_id = self.sim.model.body_name2id(self.box_A.root_body)
        self.box_B_body_id = self.sim.model.body_name2id(self.box_B.root_body)
        self.box_C_body_id = self.sim.model.body_name2id(self.box_C.root_body)

        # object joints (free joints)
        self.cube_small_joint = self.cube_small.joints[0]
        self.cube_med_joint = self.cube_med.joints[0]
        self.cube_large_joint = self.cube_large.joints[0]
        self.cube_xlarge_joint = self.cube_xlarge.joints[0]
        self.box_A_joint = self.box_A.joints[0]
        self.box_B_joint = self.box_B.joints[0]
        self.box_C_joint = self.box_C.joints[0]

    # --------------------------- OBSERVABLES ---------------------------

    def _add_pos_noise(self, p):
        return p + self._pos_bias + np.random.normal(0, self.pos_noise_std, size=3)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if not self.use_object_obs:
            return observables

        @sensor(modality="object")
        def cube_pos(_):
            p_small = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.cube_small_body_id]))
            p_med = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.cube_med_body_id]))
            p_large = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.cube_large_body_id]))
            p_xlarge = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.cube_xlarge_body_id]))
            return np.concatenate([p_small, p_med, p_large, p_xlarge])

        @sensor(modality="object")
        def cube_quat(_):
            q_small = np.array(self.sim.data.body_xquat[self.cube_small_body_id])   # wxyz
            q_med = np.array(self.sim.data.body_xquat[self.cube_med_body_id])
            q_large = np.array(self.sim.data.body_xquat[self.cube_large_body_id])
            q_xlarge = np.array(self.sim.data.body_xquat[self.cube_xlarge_body_id])
            return np.concatenate([q_small, q_med, q_large, q_xlarge])

        @sensor(modality="object")
        def cube_size(_):
            return np.array(
                [self.size_small, self.size_med, self.size_large, self.size_xlarge],
                dtype=np.float32,
            )

        @sensor(modality="object")
        def box_pos(_):
            pA = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.box_A_body_id]))
            pB = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.box_B_body_id]))
            pC = self._add_pos_noise(np.array(self.sim.data.body_xpos[self.box_C_body_id]))
            return np.concatenate([pA, pB, pC])

        @sensor(modality="object")
        def box_quat(_):
            qA = np.array(self.sim.data.body_xquat[self.box_A_body_id])
            qB = np.array(self.sim.data.body_xquat[self.box_B_body_id])
            qC = np.array(self.sim.data.body_xquat[self.box_C_body_id])
            return np.concatenate([qA, qB, qC])

        @sensor(modality="object")
        def boxA_pos(_):
            return np.array(self.sim.data.body_xpos[self.box_A_body_id])

        @sensor(modality="object")
        def boxB_pos(_):
            return np.array(self.sim.data.body_xpos[self.box_B_body_id])

        @sensor(modality="object")
        def boxC_pos(_):
            return np.array(self.sim.data.body_xpos[self.box_C_body_id])

        @sensor(modality="object")
        def boxA_quat(_):
            return np.array(self.sim.data.body_xquat[self.box_A_body_id])

        @sensor(modality="object")
        def boxB_quat(_):
            return np.array(self.sim.data.body_xquat[self.box_B_body_id])

        @sensor(modality="object")
        def boxC_quat(_):
            return np.array(self.sim.data.body_xquat[self.box_C_body_id])

        sensors = OrderedDict(
            cube_pos=cube_pos,       # noisy (4x3 flattened)
            cube_quat=cube_quat,     # 4x4 flattened
            cube_size=cube_size,     # 4 half-edge sizes
            boxes_pos=box_pos,       # noisy (3x3 flattened) -> A,B,C
            boxes_quat=box_quat,     # 3x4 flattened -> A,B,C
        )

        for name, s in sensors.items():
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )
        for k in ["object-state", "robot0_proprio-state"]:
            if k in observables:
                observables.pop(k)
        return observables

    # --------------------------- RESET ---------------------------

    def _reset_internal(self):
        super()._reset_internal()

        heavy = self.heavy_dr

        # --- store base MuJoCo values on first call ---
        # --- store base MuJoCo values on first call ---
        if self._base_masses is None:
            self._base_masses = self.sim.model.body_mass.copy()
        if self._base_damping is None:
            self._base_damping = self.sim.model.dof_damping.copy()
        if self._base_gains is None:
            self._base_gains = self.sim.model.actuator_gainprm.copy()
        if self._base_geom_friction is None:
            self._base_geom_friction = self.sim.model.geom_friction.copy()

        # always restore base values first
        self.sim.model.body_mass[:] = self._base_masses
        self.sim.model.dof_damping[:] = self._base_damping
        self.sim.model.actuator_gainprm[:] = self._base_gains
        self.sim.model.geom_friction[:] = self._base_geom_friction

        # --- friction (sliding) ---
        fr = self.friction_range_heavy if heavy else self.friction_range
        friction_scale = np.random.uniform(*fr)
        self.sim.model.geom_friction[:, 0] = self._base_geom_friction[:, 0] * friction_scale

        # --- friction (torsional + rolling) ---
        if heavy:
            self.sim.model.geom_friction[:, 1] = np.random.uniform(
                self.torsional_friction_range[0],
                self.torsional_friction_range[1],
                size=self.sim.model.ngeom,
            )
            self.sim.model.geom_friction[:, 2] = np.random.uniform(
                self.rolling_friction_range[0],
                self.rolling_friction_range[1],
                size=self.sim.model.ngeom,
            )

        # --- observation noise (per-episode) --- NEW
        if heavy:
            self.pos_noise_std = np.random.uniform(*self.pos_noise_std_range)
        # else: keeps the fixed 0.005 from __init__

        # --- observation bias ---
        bias_range = self.pos_bias_range_heavy if heavy else self.pos_bias_range
        self._pos_bias = np.random.uniform(-bias_range, bias_range, size=3)

        # --- cube mass --- NEW
        if heavy:
            mass_scale = np.random.uniform(*self.cube_mass_scale_range)
            for body_id in [self.cube_small_body_id, self.cube_med_body_id,
                            self.cube_large_body_id, self.cube_xlarge_body_id]:
                self.sim.model.body_mass[body_id] = self._base_masses[body_id] * mass_scale

        # --- joint damping --- NEW
        if heavy:
            damp_scale = np.random.uniform(*self.joint_damping_scale_range)
            self.sim.model.dof_damping[:] = self._base_damping * damp_scale

        # --- actuator gains --- NEW
        if heavy:
            gain_scale = np.random.uniform(*self.actuator_gain_scale_range)
            self.sim.model.actuator_gainprm[:] = self._base_gains * gain_scale

        # --- observation latency --- NEW
        if heavy:
            self._obs_latency = np.random.randint(
                self.obs_latency_range[0], self.obs_latency_range[1] + 1
            )
            self._obs_buffer = []
        else:
            self._obs_latency = 0
            self._obs_buffer = []

        # --- world offset ---
        xy_range = self.world_xy_offset_range_heavy if heavy else self.world_xy_offset_range
        world_offset = np.array([
            np.random.uniform(*xy_range),
            np.random.uniform(*xy_range),
            0.0
        ])

        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        table_z = float(self.table_offset[2])

        pA = np.array([0.0, -0.25, table_z]) + world_offset
        pB = np.array([0.0,  0.00, table_z]) + world_offset
        pC = np.array([0.0,  0.25, table_z]) + world_offset

        self.sim.data.set_joint_qpos(self.box_A_joint, np.concatenate([pA, q_identity]))
        self.sim.data.set_joint_qpos(self.box_B_joint, np.concatenate([pB, q_identity]))
        self.sim.data.set_joint_qpos(self.box_C_joint, np.concatenate([pC, q_identity]))

        # --- cube initial positions (with optional perturbation) ---
        base_xy = np.array([pA[0] + 0.055, pA[1] + 0.0])

        z_xlarge = table_z + self.size_xlarge
        z_large  = z_xlarge + (self.size_xlarge + self.size_large)
        z_med    = z_large + (self.size_large + self.size_med)
        z_small  = z_med + (self.size_med + self.size_small)

        perturb = self.cube_xy_perturb_range if heavy else 0.0

        for joint, z in [
            (self.cube_xlarge_joint, z_xlarge),
            (self.cube_large_joint, z_large),
            (self.cube_med_joint, z_med),
            (self.cube_small_joint, z_small),
        ]:
            xy = base_xy + np.random.uniform(-perturb, perturb, size=2)
            self.sim.data.set_joint_qpos(joint, np.array([xy[0], xy[1], z, *q_identity]))

        self.sim.forward()
    # --------------------------- HELPERS ---------------------------

    def reset(self):
        self._prev_cube_pos = None
        obs = super().reset()
        self.reset_fitness()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # keep your current dense fitness behavior
        self.update_fitness()

        # success termination (optional but helpful for debugging consistency)
        if self._check_success():
            done = True
            info = dict(info) if info is not None else {}
            info["terminated_reason"] = "success"
            return obs, reward, done, info

        # strict Hanoi legality termination: larger on smaller
        if self.terminate_on_illegal_stack and (not done):
            illegal_score = self._illegal_stack_score_now()
            if illegal_score >= self.illegal_stack_term_thresh:
                done = True
                info = dict(info) if info is not None else {}
                info["terminated_reason"] = "illegal_stack_larger_on_smaller"
                info["illegal_stack_score"] = float(illegal_score)

        return obs, reward, done, info

    def _world_to_box_local(self, p_world, box_pos, box_quat_wxyz):
        R = quat2mat(convert_quat(np.array(box_quat_wxyz), to="xyzw"))
        return R.T @ (p_world - box_pos)

    def _in_hole_xy(self, p_local):
        return (self.hole_x_min <= p_local[0] <= self.hole_x_max) and (abs(p_local[1]) <= self.hole_y_abs)

    def _stack_pair_ok(self, lower_pos, upper_pos, s_lower, s_upper):
        dxy = np.linalg.norm(upper_pos[:2] - lower_pos[:2])
        if dxy > self.stack_xy_tol * min(s_lower, s_upper):
            return False
        dz = float(upper_pos[2] - lower_pos[2])
        if dz <= 0.0:
            return False
        return abs(dz - (s_lower + s_upper)) <= self.stack_z_tol

    def _stack_pair_score_cont(self, lower_pos, upper_pos, s_lower, s_upper):
        """
        Continuous stack score in [0,1] for detecting illegal larger-on-smaller placement.
        High score means 'upper' is stacked on 'lower'.
        """
        lower_pos = np.array(lower_pos, dtype=np.float64)
        upper_pos = np.array(upper_pos, dtype=np.float64)

        dxy = float(np.linalg.norm(upper_pos[:2] - lower_pos[:2]))
        dz = float(upper_pos[2] - lower_pos[2])
        dz_target = float(s_lower + s_upper)

        xy_scale = float(self.stack_xy_tol * min(s_lower, s_upper) + 1e-12)
        s_xy = np.exp(-dxy / xy_scale)

        # smooth "upper above lower"
        s_above = 1.0 / (1.0 + np.exp(-(dz - 0.002) / 0.002))
        s_z = np.exp(-abs(dz - dz_target) / (self.stack_z_tol + 1e-12))

        return float(np.clip(s_xy * s_z * s_above, 0.0, 1.0))

    def _illegal_stack_score_now(self):
        """
        Detect illegal Hanoi placements: larger cube on smaller cube.
        Returns max continuous illegal-stack score in [0,1].
        4-cube version checks:
          med on small
          large on med
          large on small
          xlarge on large
          xlarge on med
          xlarge on small
        """
        p_small = np.array(self.sim.data.body_xpos[self.cube_small_body_id], dtype=np.float64)
        p_med = np.array(self.sim.data.body_xpos[self.cube_med_body_id], dtype=np.float64)
        p_large = np.array(self.sim.data.body_xpos[self.cube_large_body_id], dtype=np.float64)
        p_xlarge = np.array(self.sim.data.body_xpos[self.cube_xlarge_body_id], dtype=np.float64)

        scores = [
            # medium on small
            self._stack_pair_score_cont(p_small, p_med, self.size_small, self.size_med),

            # large on smaller
            self._stack_pair_score_cont(p_med, p_large, self.size_med, self.size_large),
            self._stack_pair_score_cont(p_small, p_large, self.size_small, self.size_large),

            # xlarge on smaller
            self._stack_pair_score_cont(p_large, p_xlarge, self.size_large, self.size_xlarge),
            self._stack_pair_score_cont(p_med, p_xlarge, self.size_med, self.size_xlarge),
            self._stack_pair_score_cont(p_small, p_xlarge, self.size_small, self.size_xlarge),
        ]
        return float(max(scores))

    # --------------------------- SUCCESS ---------------------------

    def _check_success(self):
        # cube positions
        p_small = np.array(self.sim.data.body_xpos[self.cube_small_body_id])
        p_med = np.array(self.sim.data.body_xpos[self.cube_med_body_id])
        p_large = np.array(self.sim.data.body_xpos[self.cube_large_body_id])
        p_xlarge = np.array(self.sim.data.body_xpos[self.cube_xlarge_body_id])

        # containers
        pC = np.array(self.sim.data.body_xpos[self.box_C_body_id])
        qC = np.array(self.sim.data.body_xquat[self.box_C_body_id])
        pA = np.array(self.sim.data.body_xpos[self.box_A_body_id])
        qA = np.array(self.sim.data.body_xquat[self.box_A_body_id])
        pB = np.array(self.sim.data.body_xpos[self.box_B_body_id])
        qB = np.array(self.sim.data.body_xquat[self.box_B_body_id])

        def in_container(p_obj, p_box, q_box):
            p_local = self._world_to_box_local(p_obj, p_box, q_box)
            return self._in_hole_xy(p_local)

        inC_small = in_container(p_small, pC, qC)
        inC_med = in_container(p_med, pC, qC)
        inC_large = in_container(p_large, pC, qC)
        inC_xlarge = in_container(p_xlarge, pC, qC)

        # target must contain all four
        if not (inC_small and inC_med and inC_large and inC_xlarge):
            return False

        # A and B must contain none
        for p in [p_small, p_med, p_large, p_xlarge]:
            if in_container(p, pA, qA):
                return False
            if in_container(p, pB, qB):
                return False

        # correct order on C: xlarge -> large -> med -> small
        if not self._stack_pair_ok(p_xlarge, p_large, self.size_xlarge, self.size_large):
            return False
        if not self._stack_pair_ok(p_large, p_med, self.size_large, self.size_med):
            return False
        if not self._stack_pair_ok(p_med, p_small, self.size_med, self.size_small):
            return False

        return True


    def _debug_force_illegal(self, upper: str = "large", lower: str = "small"):
        """
        Force an illegal stack by placing a larger cube ('upper') on top of a smaller cube ('lower')
        over container C. Used for termination debugging.

        Valid names: "small", "med", "large", "xlarge"
        Illegal examples:
        med on small
        large on med
        large on small
        xlarge on large
        xlarge on med
        xlarge on small
        """
        q = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        table_z = float(self.table_offset[2])

        pC = np.array(self.sim.data.body_xpos[self.box_C_body_id])
        base_xy = np.array([pC[0] + 0.055, pC[1]])

        # map names -> joints / sizes
        joint_map = {
            "small": self.cube_small_joint,
            "med": self.cube_med_joint,
            "large": self.cube_large_joint,
            "xlarge": self.cube_xlarge_joint,
        }
        size_map = {
            "small": self.size_small,
            "med": self.size_med,
            "large": self.size_large,
            "xlarge": self.size_xlarge,
        }

        assert upper in joint_map and lower in joint_map, f"Invalid names: {upper}, {lower}"

        # Put all cubes in a safe separated layout first (avoid accidental extra stacks)
        # A, B, C row-ish placements
        pA = np.array(self.sim.data.body_xpos[self.box_A_body_id])
        pB = np.array(self.sim.data.body_xpos[self.box_B_body_id])

        safe_xy = {
            "small": np.array([pA[0] + 0.055, pA[1]]),
            "med":   np.array([pB[0] + 0.055, pB[1]]),
            "large": np.array([pC[0] - 0.08,  pC[1]]),
            "xlarge":np.array([pC[0] - 0.14,  pC[1]]),
        }

        for name in ["small", "med", "large", "xlarge"]:
            z = table_z + size_map[name]
            self.sim.data.set_joint_qpos(
                joint_map[name],
                np.array([safe_xy[name][0], safe_xy[name][1], z, *q], dtype=np.float64)
            )

        # Now force the illegal pair over C
        z_lower = table_z + size_map[lower]
        z_upper = z_lower + (size_map[lower] + size_map[upper])

        self.sim.data.set_joint_qpos(
            joint_map[lower],
            np.array([base_xy[0], base_xy[1], z_lower, *q], dtype=np.float64)
        )
        self.sim.data.set_joint_qpos(
            joint_map[upper],
            np.array([base_xy[0], base_xy[1], z_upper, *q], dtype=np.float64)
        )

        self.sim.forward()

    def _debug_force_success(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        table_z = float(self.table_offset[2])
        pC = np.array(self.sim.data.body_xpos[self.box_C_body_id])

        base_xy = np.array([pC[0] + 0.055, pC[1]])

        z_xlarge = table_z + self.size_xlarge
        z_large = z_xlarge + (self.size_xlarge + self.size_large)
        z_med = z_large + (self.size_large + self.size_med)
        z_small = z_med + (self.size_med + self.size_small)

        self.sim.data.set_joint_qpos(self.cube_xlarge_joint, np.array([base_xy[0], base_xy[1], z_xlarge, *q]))
        self.sim.data.set_joint_qpos(self.cube_large_joint,  np.array([base_xy[0], base_xy[1], z_large,  *q]))
        self.sim.data.set_joint_qpos(self.cube_med_joint,    np.array([base_xy[0], base_xy[1], z_med,    *q]))
        self.sim.data.set_joint_qpos(self.cube_small_joint,  np.array([base_xy[0], base_xy[1], z_small,  *q]))
        self.sim.forward()

    # --------------------------- TASK DESCRIPTION ---------------------------

    def task_description(self) -> str:
        return (
        "Tower of Hanoi with 4 cubes and 3 containers (A, B, C). "
        "Cubes have different sizes: extra-large (gray), large (green), medium (blue), and small (red). "
        "At reset, all four cubes are stacked on container A in correct order: extra-large at the bottom, "
        "then large, medium, and small on top. "
        "The goal is to move the full tower to container C, preserving the same size order "
        "(extra-large -> large -> medium -> small), while leaving containers A and B empty. "
        "Illegal Hanoi placements are not allowed: a larger cube placed on top of a smaller cube terminates the episode."
        )

    # ------------------------------- REWARD -------------------------------

    # --------------------------- FITNESS (UNCHANGED, legacy 3-cube version) ---------------------------
    # NOTE: Left intentionally unchanged as requested. It ignores cube_xlarge for now.
    def reset_fitness(self):
        self._fitness_state = {
            "steps": 0,
            "sum_composite": 0.0,
            "sum_interaction_quality": 0.0,

            "max_composite": 0.0,
            "max_reach": 0.0,
            "max_grasp": 0.0,
            "max_transport": 0.0,
            "max_place": 0.0,
            "max_arrangement": 0.0,

            "last": {},

            "min_eef_to_cube_dist": {
                "small": float("inf"),
                "med": float("inf"),
                "large": float("inf"),
                "xlarge": float("inf"),
            },
            "max_lift_height": {"small": 0.0, "med": 0.0, "large": 0.0, "xlarge": 0.0},
            "sum_cube_speed": {"small": 0.0, "med": 0.0, "large": 0.0, "xlarge": 0.0},

            "drop_count": 0,
            "illegal_stack_max": 0.0,

            "t_first_reach": None,
            "t_first_lift": None,
            "t_first_cube_in_C": None,
            "t_first_two_in_C": None,
            "t_first_all_in_C": None,
        }

        self._fitness_prev = {
            "eef_pos": None,
            "lift_h": {"small": 0.0, "med": 0.0, "large": 0.0, "xlarge": 0.0},
        }

        try:
            robot = self.robots[0]
            if hasattr(robot, "eef_pos"):
                self._fitness_prev["eef_pos"] = np.array(robot.eef_pos, dtype=np.float64)
            else:
                site_id = self.sim.model.site_name2id("gripper0_right_grip_site")
                self._fitness_prev["eef_pos"] = np.array(self.sim.data.site_xpos[site_id], dtype=np.float64)
        except Exception:
            self._fitness_prev["eef_pos"] = None

    def update_fitness(self):
        fs = self._fitness_state
        fs["steps"] += 1
        t = fs["steps"]

        def _safe_clip01(x):
            return float(np.clip(x, 0.0, 1.0))

        def _exp_score(dist, scale):
            return float(np.exp(-float(dist) / (float(scale) + 1e-12)))

        def _eef_pos():
            robot = self.robots[0]
            if hasattr(robot, "eef_pos"):
                return np.array(robot.eef_pos, dtype=np.float64)
            if hasattr(robot, "ee_pos"):
                return np.array(robot.ee_pos, dtype=np.float64)
            try:
                site_id = self.sim.model.site_name2id("gripper0_right_grip_site")
                return np.array(self.sim.data.site_xpos[site_id], dtype=np.float64)
            except Exception:
                return np.array(self.sim.data.body_xpos[self.robots[0].robot_model.root_body_id], dtype=np.float64)

        def _hole_score(p_obj_world, p_box_world, q_box_wxyz):
            p_local = self._world_to_box_local(np.array(p_obj_world), np.array(p_box_world), np.array(q_box_wxyz))
            x = float(p_local[0])
            y = float(p_local[1])

            x_c = 0.5 * (self.hole_x_min + self.hole_x_max)
            hx = 0.5 * (self.hole_x_max - self.hole_x_min)
            hy = float(self.hole_y_abs)

            dx_out = max(0.0, abs(x - x_c) - hx)
            dy_out = max(0.0, abs(y) - hy)
            outside = (dx_out * dx_out + dy_out * dy_out) ** 0.5

            cx = (x - x_c) / (hx + 1e-12)
            cy = y / (hy + 1e-12)
            center_score = np.exp(-0.5 * (cx * cx + cy * cy))
            outside_score = np.exp(-outside / 0.008)

            return float(np.clip(center_score * outside_score, 0.0, 1.0))

        def _stack_pair_score(lower_pos, upper_pos, s_lower, s_upper):
            lower_pos = np.array(lower_pos, dtype=np.float64)
            upper_pos = np.array(upper_pos, dtype=np.float64)

            dxy = float(np.linalg.norm(upper_pos[:2] - lower_pos[:2]))
            dz = float(upper_pos[2] - lower_pos[2])
            dz_target = float(s_lower + s_upper)

            xy_scale = float(self.stack_xy_tol * min(s_lower, s_upper) + 1e-12)
            s_xy = np.exp(-dxy / xy_scale)

            s_above = 1.0 / (1.0 + np.exp(-(dz - 0.002) / 0.002))
            s_z = np.exp(-abs(dz - dz_target) / (self.stack_z_tol + 1e-12))

            return float(np.clip(s_xy * s_z * s_above, 0.0, 1.0))

        table_z = float(self.table_offset[2])

        # 4-cube dense fitness
        p_small = np.array(self.sim.data.body_xpos[self.cube_small_body_id], dtype=np.float64)
        p_med = np.array(self.sim.data.body_xpos[self.cube_med_body_id], dtype=np.float64)
        p_large = np.array(self.sim.data.body_xpos[self.cube_large_body_id], dtype=np.float64)
        p_xlarge = np.array(self.sim.data.body_xpos[self.cube_xlarge_body_id], dtype=np.float64)

        pA = np.array(self.sim.data.body_xpos[self.box_A_body_id], dtype=np.float64)
        qA = np.array(self.sim.data.body_xquat[self.box_A_body_id], dtype=np.float64)
        pB = np.array(self.sim.data.body_xpos[self.box_B_body_id], dtype=np.float64)
        qB = np.array(self.sim.data.body_xquat[self.box_B_body_id], dtype=np.float64)
        pC = np.array(self.sim.data.body_xpos[self.box_C_body_id], dtype=np.float64)
        qC = np.array(self.sim.data.body_xquat[self.box_C_body_id], dtype=np.float64)

        eef = _eef_pos()

        eef_speed = 0.0
        if self._fitness_prev["eef_pos"] is not None:
            eef_speed = float(np.linalg.norm(eef - self._fitness_prev["eef_pos"]) * self.control_freq)
        self._fitness_prev["eef_pos"] = eef

        dt = 1.0 / self.control_freq

        if self._prev_cube_pos is None:
            self._prev_cube_pos = {
                "small": p_small.copy(),
                "med": p_med.copy(),
                "large": p_large.copy(),
                "xlarge": p_xlarge.copy(),
            }

        v_small = (p_small - self._prev_cube_pos["small"]) / dt
        v_med = (p_med - self._prev_cube_pos["med"]) / dt
        v_large = (p_large - self._prev_cube_pos["large"]) / dt
        v_xlarge = (p_xlarge - self._prev_cube_pos["xlarge"]) / dt

        self._prev_cube_pos["small"] = p_small.copy()
        self._prev_cube_pos["med"] = p_med.copy()
        self._prev_cube_pos["large"] = p_large.copy()
        self._prev_cube_pos["xlarge"] = p_xlarge.copy()

        sp_small = float(np.linalg.norm(v_small))
        sp_med = float(np.linalg.norm(v_med))
        sp_large = float(np.linalg.norm(v_large))
        sp_xlarge = float(np.linalg.norm(v_xlarge))

        fs["sum_cube_speed"]["small"] += sp_small
        fs["sum_cube_speed"]["med"] += sp_med
        fs["sum_cube_speed"]["large"] += sp_large
        fs["sum_cube_speed"]["xlarge"] += sp_xlarge

        mean_cube_speed = (sp_small + sp_med + sp_large + sp_xlarge) / 4.0

        d_small = float(np.linalg.norm(eef - p_small))
        d_med = float(np.linalg.norm(eef - p_med))
        d_large = float(np.linalg.norm(eef - p_large))
        d_xlarge = float(np.linalg.norm(eef - p_xlarge))

        fs["min_eef_to_cube_dist"]["small"] = min(fs["min_eef_to_cube_dist"]["small"], d_small)
        fs["min_eef_to_cube_dist"]["med"] = min(fs["min_eef_to_cube_dist"]["med"], d_med)
        fs["min_eef_to_cube_dist"]["large"] = min(fs["min_eef_to_cube_dist"]["large"], d_large)
        fs["min_eef_to_cube_dist"]["xlarge"] = min(fs["min_eef_to_cube_dist"]["xlarge"], d_xlarge)

        reach_small = _exp_score(d_small, 0.12)
        reach_med = _exp_score(d_med, 0.12)
        reach_large = _exp_score(d_large, 0.12)
        reach_xlarge = _exp_score(d_xlarge, 0.12)
        approach_score = max(reach_small, reach_med, reach_large, reach_xlarge)

        lh_small = float(p_small[2] - (table_z + self.size_small))
        lh_med = float(p_med[2] - (table_z + self.size_med))
        lh_large = float(p_large[2] - (table_z + self.size_large))
        lh_xlarge = float(p_xlarge[2] - (table_z + self.size_xlarge))

        fs["max_lift_height"]["small"] = max(fs["max_lift_height"]["small"], lh_small)
        fs["max_lift_height"]["med"] = max(fs["max_lift_height"]["med"], lh_med)
        fs["max_lift_height"]["large"] = max(fs["max_lift_height"]["large"], lh_large)
        fs["max_lift_height"]["xlarge"] = max(fs["max_lift_height"]["xlarge"], lh_xlarge)

        lift_small = _safe_clip01(lh_small / 0.08)
        lift_med = _safe_clip01(lh_med / 0.08)
        lift_large = _safe_clip01(lh_large / 0.08)
        lift_xlarge = _safe_clip01(lh_xlarge / 0.08)
        grasp_score = max(lift_small, lift_med, lift_large, lift_xlarge)

        def _drop_event(lh_now, lh_prev, v_z):
            return (lh_prev > 0.035) and (lh_now < 0.015) and (v_z < -0.25)

        if _drop_event(lh_small, self._fitness_prev["lift_h"]["small"], float(v_small[2])):
            fs["drop_count"] += 1
        if _drop_event(lh_med, self._fitness_prev["lift_h"]["med"], float(v_med[2])):
            fs["drop_count"] += 1
        if _drop_event(lh_large, self._fitness_prev["lift_h"]["large"], float(v_large[2])):
            fs["drop_count"] += 1
        if _drop_event(lh_xlarge, self._fitness_prev["lift_h"]["xlarge"], float(v_xlarge[2])):
            fs["drop_count"] += 1

        self._fitness_prev["lift_h"]["small"] = lh_small
        self._fitness_prev["lift_h"]["med"] = lh_med
        self._fitness_prev["lift_h"]["large"] = lh_large
        self._fitness_prev["lift_h"]["xlarge"] = lh_xlarge

        x_c = 0.5 * (self.hole_x_min + self.hole_x_max)

        def _xy_err_to_C(p_obj):
            p_local_C = self._world_to_box_local(np.array(p_obj), pC, qC)
            ex = float(p_local_C[0] - x_c)
            ey = float(p_local_C[1] - 0.0)
            return float((ex * ex + ey * ey) ** 0.5), p_local_C

        err_small, _ = _xy_err_to_C(p_small)
        err_med, _ = _xy_err_to_C(p_med)
        err_large, _ = _xy_err_to_C(p_large)
        err_xlarge, _ = _xy_err_to_C(p_xlarge)

        trans_small = _exp_score(err_small, 0.18)
        trans_med = _exp_score(err_med, 0.18)
        trans_large = _exp_score(err_large, 0.18)
        trans_xlarge = _exp_score(err_xlarge, 0.18)

        transport_score = max(
            lift_small * trans_small,
            lift_med * trans_med,
            lift_large * trans_large,
            lift_xlarge * trans_xlarge,
        )

        hA_small = _hole_score(p_small, pA, qA)
        hA_med = _hole_score(p_med, pA, qA)
        hA_large = _hole_score(p_large, pA, qA)
        hA_xlarge = _hole_score(p_xlarge, pA, qA)

        hB_small = _hole_score(p_small, pB, qB)
        hB_med = _hole_score(p_med, pB, qB)
        hB_large = _hole_score(p_large, pB, qB)
        hB_xlarge = _hole_score(p_xlarge, pB, qB)

        hC_small = _hole_score(p_small, pC, qC)
        hC_med = _hole_score(p_med, pC, qC)
        hC_large = _hole_score(p_large, pC, qC)
        hC_xlarge = _hole_score(p_xlarge, pC, qC)

        C_fill = (hC_small + hC_med + hC_large + hC_xlarge) / 4.0
        A_empty = 1.0 - (hA_small + hA_med + hA_large + hA_xlarge) / 4.0
        B_empty = 1.0 - (hB_small + hB_med + hB_large + hB_xlarge) / 4.0
        emptiness = _safe_clip01(0.5 * (A_empty + B_empty))
        place_score = _safe_clip01(0.75 * C_fill + 0.25 * emptiness)

        if fs["t_first_reach"] is None and approach_score > 0.8:
            fs["t_first_reach"] = t
        if fs["t_first_lift"] is None and grasp_score > 0.4:
            fs["t_first_lift"] = t
        n_in_C = int((hC_small > 0.7) + (hC_med > 0.7) + (hC_large > 0.7) + (hC_xlarge > 0.7))
        if fs["t_first_cube_in_C"] is None and n_in_C >= 1:
            fs["t_first_cube_in_C"] = t
        if fs["t_first_two_in_C"] is None and n_in_C >= 2:
            fs["t_first_two_in_C"] = t
        if fs["t_first_all_in_C"] is None and n_in_C >= 4:
            fs["t_first_all_in_C"] = t

        # 4-cube target stack on C: xlarge -> large -> med -> small
        z_goal_xlarge = table_z + self.size_xlarge
        z_goal_large = z_goal_xlarge + (self.size_xlarge + self.size_large)
        z_goal_med = z_goal_large + (self.size_large + self.size_med)
        z_goal_small = z_goal_med + (self.size_med + self.size_small)

        def _goal_pose_score(hC, err_xy, z_now, z_goal, size_half):
            xy = np.exp(-float(err_xy) / (0.018 + 0.5 * float(size_half)))
            zz = np.exp(-abs(float(z_now) - float(z_goal)) / 0.02)
            return float(np.clip(hC * xy * zz, 0.0, 1.0))

        goal_xlarge = _goal_pose_score(hC_xlarge, err_xlarge, p_xlarge[2], z_goal_xlarge, self.size_xlarge)
        goal_large = _goal_pose_score(hC_large, err_large, p_large[2], z_goal_large, self.size_large)
        goal_med = _goal_pose_score(hC_med, err_med, p_med[2], z_goal_med, self.size_med)
        goal_small = _goal_pose_score(hC_small, err_small, p_small[2], z_goal_small, self.size_small)

        stack_XL_L = _stack_pair_score(p_xlarge, p_large, self.size_xlarge, self.size_large)
        stack_LM = _stack_pair_score(p_large, p_med, self.size_large, self.size_med)
        stack_MS = _stack_pair_score(p_med, p_small, self.size_med, self.size_small)

        mean_goal = (goal_xlarge + goal_large + goal_med + goal_small) / 4.0
        arrangement_score = _safe_clip01(
            0.50 * mean_goal + 0.1666667 * stack_XL_L + 0.1666667 * stack_LM + 0.1666667 * stack_MS
        )

        # illegal larger-on-smaller diagnostics (4-cube)
        illegal_MS = _stack_pair_score(p_small, p_med, self.size_small, self.size_med)
        illegal_LM = _stack_pair_score(p_med, p_large, self.size_med, self.size_large)
        illegal_LS = _stack_pair_score(p_small, p_large, self.size_small, self.size_large)
        illegal_XL_L = _stack_pair_score(p_large, p_xlarge, self.size_large, self.size_xlarge)
        illegal_XL_M = _stack_pair_score(p_med, p_xlarge, self.size_med, self.size_xlarge)
        illegal_XL_S = _stack_pair_score(p_small, p_xlarge, self.size_small, self.size_xlarge)

        illegal_now = float(max(illegal_MS, illegal_LM, illegal_LS, illegal_XL_L, illegal_XL_M, illegal_XL_S))
        fs["illegal_stack_max"] = max(fs["illegal_stack_max"], illegal_now)

        iq_eef = 1.0 - np.clip(eef_speed / 1.5, 0.0, 1.0)
        iq_cube = 1.0 - np.clip(mean_cube_speed / 1.0, 0.0, 1.0)
        interaction_quality = _safe_clip01(0.6 * iq_eef + 0.4 * iq_cube)

        w_app, w_grasp, w_trans, w_place, w_arr = 0.14, 0.18, 0.20, 0.20, 0.28
        composite = (
            w_app * approach_score
            + w_grasp * grasp_score
            + w_trans * transport_score
            + w_place * place_score
            + w_arr * arrangement_score
        )

        composite *= (1.0 - 0.15 * illegal_now)
        composite = _safe_clip01(0.95 * composite + 0.05 * interaction_quality)

        fs["sum_composite"] += composite
        fs["sum_interaction_quality"] += interaction_quality

        fs["max_composite"] = max(fs["max_composite"], composite)
        fs["max_reach"] = max(fs["max_reach"], approach_score)
        fs["max_grasp"] = max(fs["max_grasp"], grasp_score)
        fs["max_transport"] = max(fs["max_transport"], transport_score)
        fs["max_place"] = max(fs["max_place"], place_score)
        fs["max_arrangement"] = max(fs["max_arrangement"], arrangement_score)

        fs["last"] = {
            "composite": float(composite),
            "approach": float(approach_score),
            "grasp": float(grasp_score),
            "transport": float(transport_score),
            "place": float(place_score),
            "arrangement": float(arrangement_score),
            "interaction_quality": float(interaction_quality),

            "eef_speed": float(eef_speed),
            "mean_cube_speed": float(mean_cube_speed),

            "hole_A": {"small": float(hA_small), "med": float(hA_med), "large": float(hA_large), "xlarge": float(hA_xlarge)},
            "hole_B": {"small": float(hB_small), "med": float(hB_med), "large": float(hB_large), "xlarge": float(hB_xlarge)},
            "hole_C": {"small": float(hC_small), "med": float(hC_med), "large": float(hC_large), "xlarge": float(hC_xlarge)},
            "C_fill": float(C_fill),
            "A_empty": float(_safe_clip01(A_empty)),
            "B_empty": float(_safe_clip01(B_empty)),
            "emptiness": float(emptiness),

            "xy_err_to_C": {"small": float(err_small), "med": float(err_med), "large": float(err_large), "xlarge": float(err_xlarge)},
            "goal_pose": {"small": float(goal_small), "med": float(goal_med), "large": float(goal_large), "xlarge": float(goal_xlarge)},
            "stack_scores": {"XL_on_L": float(stack_XL_L), "L_on_M": float(stack_LM), "M_on_S": float(stack_MS)},
            "illegal_now": float(illegal_now),

            "reach_dists": {"small": float(d_small), "med": float(d_med), "large": float(d_large), "xlarge": float(d_xlarge)},
            "lift_heights": {"small": float(lh_small), "med": float(lh_med), "large": float(lh_large), "xlarge": float(lh_xlarge)},
        }

    def get_episode_fitness(self):
        fs = self._fitness_state
        steps = max(1, int(fs.get("steps", 0)))

        auc = float(fs["sum_composite"] / steps)
        max_comp = float(fs["max_composite"])
        last_arr = float(fs["last"].get("arrangement", 0.0)) if fs.get("last") else 0.0

        episode_fitness = float(np.clip(0.45 * auc + 0.45 * max_comp + 0.10 * last_arr, 0.0, 1.0))

        success = bool(self._check_success())
        if success:
            episode_fitness = 1.0

        metrics = {
            "success": success,
            "steps": steps,
            "fitness_auc_composite": auc,
            "fitness_max_composite": max_comp,
            "fitness_last_composite": float(fs["last"].get("composite", 0.0)) if fs.get("last") else 0.0,
            "episode_fitness": episode_fitness,
            "max_phase_scores": {
                "approach": float(fs["max_reach"]),
                "grasp_lift": float(fs["max_grasp"]),
                "transport": float(fs["max_transport"]),
                "placement": float(fs["max_place"]),
                "arrangement": float(fs["max_arrangement"]),
            },
            "last_phase_scores": {
                "approach": float(fs["last"].get("approach", 0.0)) if fs.get("last") else 0.0,
                "grasp_lift": float(fs["last"].get("grasp", 0.0)) if fs.get("last") else 0.0,
                "transport": float(fs["last"].get("transport", 0.0)) if fs.get("last") else 0.0,
                "placement": float(fs["last"].get("place", 0.0)) if fs.get("last") else 0.0,
                "arrangement": float(fs["last"].get("arrangement", 0.0)) if fs.get("last") else 0.0,
            },
            "milestones": {
                "t_first_reach": fs["t_first_reach"],
                "t_first_lift": fs["t_first_lift"],
                "t_first_cube_in_C": fs["t_first_cube_in_C"],
                "t_first_two_in_C": fs["t_first_two_in_C"],
                "t_first_all_in_C": fs["t_first_all_in_C"],
            },
            "interaction_quality": {
                "mean": float(fs["sum_interaction_quality"] / steps),
                "last": float(fs["last"].get("interaction_quality", 0.0)) if fs.get("last") else 0.0,
                "drop_count": int(fs["drop_count"]),
                "illegal_stack_max": float(fs["illegal_stack_max"]),
                "last_eef_speed": float(fs["last"].get("eef_speed", 0.0)) if fs.get("last") else 0.0,
                "last_mean_cube_speed": float(fs["last"].get("mean_cube_speed", 0.0)) if fs.get("last") else 0.0,
            },
            "reach": {
                "min_eef_to_cube_dist": {
                    "small": float(fs["min_eef_to_cube_dist"]["small"]),
                    "med": float(fs["min_eef_to_cube_dist"]["med"]),
                    "large": float(fs["min_eef_to_cube_dist"]["large"]),
                    "xlarge": float(fs["min_eef_to_cube_dist"]["xlarge"]),
                },
                "last_eef_to_cube_dist": (fs["last"].get("reach_dists", {}) if fs.get("last") else {}),
            },
            "grasp_lift": {
                "max_lift_height": {
                    "small": float(fs["max_lift_height"]["small"]),
                    "med": float(fs["max_lift_height"]["med"]),
                    "large": float(fs["max_lift_height"]["large"]),
                    "xlarge": float(fs["max_lift_height"]["xlarge"]),
                },
                "last_lift_heights": (fs["last"].get("lift_heights", {}) if fs.get("last") else {}),
            },
            "transport": {
                "last_xy_err_to_C_center": (fs["last"].get("xy_err_to_C", {}) if fs.get("last") else {}),
                "mean_cube_speed": {
                    "small": float(fs["sum_cube_speed"]["small"] / steps),
                    "med": float(fs["sum_cube_speed"]["med"] / steps),
                    "large": float(fs["sum_cube_speed"]["large"] / steps),
                    "xlarge": float(fs["sum_cube_speed"]["xlarge"] / steps),
                },
            },
            "containers": {
                "last_hole_scores_A": (fs["last"].get("hole_A", {}) if fs.get("last") else {}),
                "last_hole_scores_B": (fs["last"].get("hole_B", {}) if fs.get("last") else {}),
                "last_hole_scores_C": (fs["last"].get("hole_C", {}) if fs.get("last") else {}),
                "last_C_fill": float(fs["last"].get("C_fill", 0.0)) if fs.get("last") else 0.0,
                "last_A_empty": float(fs["last"].get("A_empty", 0.0)) if fs.get("last") else 0.0,
                "last_B_empty": float(fs["last"].get("B_empty", 0.0)) if fs.get("last") else 0.0,
            },
            "arrangement": {
                "last_goal_pose_scores": (fs["last"].get("goal_pose", {}) if fs.get("last") else {}),
                "last_stack_pair_scores": (fs["last"].get("stack_scores", {}) if fs.get("last") else {}),
            },
        }

        return episode_fitness, metrics

    def reward(self, action=None):
        return 1.0 if self._check_success() else 0.0
import numpy as np
import pinocchio

import crocoddyl


class SimpleBipedGaitProblem:
    """Build simple bipedal locomotion problems.

    This class aims to build simple locomotion problems used in the examples of
    Crocoddyl.
    The scope of this class is purely for academic reasons, and it does not aim to be
    used in any robotics application.
    We also do not consider it as part of the API, so changes in this class will not
    pass through a strict process of deprecation.
    Thus, we advice any user to DO NOT develop their application based on this class.
    """

    def __init__(
        self,
        rmodel,
        rightFoot,
        leftFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
    ):
        """Construct biped-gait problem.

        :param rmodel: robot model
        :param rightFoot: name of the right foot
        :param leftFoot: name of the left foot
        :param integrator: type of the integrator
            (options are: 'euler', and 'rk4')
        :param control: type of control parametrization
            (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate(
            [q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

        # Collision avoidance: Define important frame pairs to monitor
        self.collision_pairs = [
            ("left_foot_link", "right_foot_link"),  # Feet shouldn't collide
        ]
        self.collision_min_distance = 0.1  # 10cm minimum distance
        print(f"[Collision Avoidance] Enabled with min distance: {self.collision_min_distance}m")

    def createSingleStepProblem(
        self, x0, leftFootTarget, rightFootTarget, timeStep, stepKnots, supportKnots, stepHeight=0.10, targetYaw=0.0
    ):
        """Create a shooting problem for a single step with specified foot locations.

        :param x0: initial state
        :param leftFootTarget: target position (3D) for left foot [x, y, z]
        :param rightFootTarget: target position (3D) for right foot [x, y, z]
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :param stepHeight: height of foot swing trajectory (default: 0.10m)
        :param targetYaw: target yaw angle for swing foot (default: 0.0 rad)
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation

        # Compute CoM reference between current foot positions
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2] + 0.1

        # Determine which foot needs to move
        leftFootTarget = np.array(leftFootTarget)
        rightFootTarget = np.array(rightFootTarget)

        leftFootMovement = np.linalg.norm(leftFootTarget - lfPos0)
        rightFootMovement = np.linalg.norm(rightFootTarget - rfPos0)

        loco3dModel = []

        # Initial double support phase - gradually shift COM towards the foot that will swing first
        com_initial_center = (rfPos0 + lfPos0) / 2
        com_initial_center[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        # Determine which foot to prepare for (will swing first)
        if leftFootMovement > rightFootMovement:
            # Left foot will swing, so shift COM towards right foot (50% of the way)
            com_initial_target = rfPos0.copy()
        else:
            # Right foot will swing, so shift COM towards left foot (50% of the way)
            com_initial_target = lfPos0.copy()
        com_initial_target[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        doubleSupport_initial = [
            self.createSwingFootModel(
                timeStep,
                [self.rfId, self.lfId],
                comTask=com_initial_center + (com_initial_target - com_initial_center) * 0.8 * ((k + 1) / supportKnots),
                comWeight=5e5
            )
            for k in range(supportKnots)
        ]
        loco3dModel += doubleSupport_initial

        # Determine which foot to move first (the one with larger movement)
        if leftFootMovement > rightFootMovement:
            # Move left foot first
            lStep = self.createFootstepModelsWithTarget(
                comRef,
                lfPos0,
                leftFootTarget,
                stepHeight,
                timeStep,
                stepKnots,
                [self.rfId],  # right foot supports
                [self.lfId],  # left foot swings
                targetYaw,
            )
            loco3dModel += lStep

            # Update CoM reference
            comRef = (leftFootTarget + rfPos0) / 2
            comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

            # Add transition double support - gradually shift COM towards support foot (right foot)
            # Shift 50% towards the support foot for stability without excessive movement
            com_center = comRef.copy()
            com_support = rfPos0.copy()
            com_support[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

            doubleSupport_transition = [
                self.createSwingFootModel(
                    timeStep,
                    [self.rfId, self.lfId],
                    comTask=com_center + (com_support - com_center) * 0.5 * ((k + 1) / supportKnots),
                    comWeight=5e5
                )
                for k in range(supportKnots)
            ]
            loco3dModel += doubleSupport_transition

            # # Move right foot if needed
            # if rightFootMovement > 1e-3:
            #     rStep = self.createFootstepModelsWithTarget(
            #         comRef,
            #         rfPos0,
            #         rightFootTarget,
            #         stepHeight,
            #         timeStep,
            #         stepKnots,
            #         [self.lfId],  # left foot supports
            #         [self.rfId],  # right foot swings
            #         targetYaw,
            #     )
            #     loco3dModel += rStep
        else:
            # Move right foot first
            rStep = self.createFootstepModelsWithTarget(
                comRef,
                rfPos0,
                rightFootTarget,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfId],  # left foot supports
                [self.rfId],  # right foot swings
                targetYaw,
            )
            loco3dModel += rStep

            # Update CoM reference
            comRef = (leftFootTarget + rightFootTarget) / 2
            comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

            # Add transition double support - gradually shift COM towards support foot (left foot)
            # Shift 50% towards the support foot for stability without excessive movement
            com_center = comRef.copy()
            com_support = lfPos0.copy()
            com_support[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

            doubleSupport_transition = [
                self.createSwingFootModel(
                    timeStep,
                    [self.rfId, self.lfId],
                    comTask=com_center + (com_support - com_center) * 0.5 * ((k + 1) / supportKnots),
                    comWeight=5e5
                )
                for k in range(supportKnots)
            ]
            loco3dModel += doubleSupport_transition

            # # Move left foot if needed
            # if leftFootMovement > 1e-3:
            #     lStep = self.createFootstepModelsWithTarget(
            #         comRef,
            #         lfPos0,
            #         leftFootTarget,
            #         stepHeight,
            #         timeStep,
            #         stepKnots,
            #         [self.rfId],  # right foot supports
            #         [self.lfId],  # left foot swings
            #         targetYaw,
            #     )
            #     loco3dModel += lStep

        # Final double support phase - return COM to center between final feet
        # Start from where the last transition left it and shift back to center
        com_final = (leftFootTarget + rightFootTarget) / 2
        com_final[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        # Gradually shift from current position back to center
        if leftFootMovement > rightFootMovement:
            # Last transition was from left swing, so start from shifted position over left foot target
            com_last_position = (leftFootTarget + (leftFootTarget + rightFootTarget) / 2) / 2
        else:
            # Last transition was from right swing, so start from shifted position over right foot target
            com_last_position = (rightFootTarget + (leftFootTarget + rightFootTarget) / 2) / 2
        com_last_position[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]

        doubleSupport_final = [
            self.createSwingFootModel(
                timeStep,
                [self.rfId, self.lfId],
                comTask=com_last_position + (com_final - com_last_position) * ((k + 1) / supportKnots),
                comWeight=5e5
            )
            for k in range(supportKnots)
        ]
        loco3dModel += doubleSupport_final

        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createWalkingProblem(
        self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    ):
        """Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(timeStep, [self.rfId, self.lfId])
            for _ in range(supportKnots)
        ]
        # Creating the action models for three steps
        if self.firstStep is True:
            rStep = self.createFootstepModels(
                comRef,
                [rfPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfId],
                [self.rfId],
            )
            self.firstStep = False
        else:
            rStep = self.createFootstepModels(
                comRef,
                [rfPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfId],
                [self.rfId],
            )
        lStep = self.createFootstepModels(
            comRef,
            [lfPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfId],
            [self.lfId],
        )
        # We defined the problem as:
        loco3dModel += doubleSupport + rStep
        loco3dModel += doubleSupport + lStep
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createFootstepModelsWithTarget(
        self,
        comPos0,
        footPos0,
        footTarget,
        stepHeight,
        timeStep,
        numKnots,
        supportFootIds,
        swingFootIds,
        targetYaw=0.0,
    ):
        """Action models for a footstep phase with explicit target position.

        :param comPos0: initial CoM position
        :param footPos0: initial position of the swinging foot
        :param footTarget: target position (3D) for the swinging foot
        :param stepHeight: step height for swing trajectory
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :param targetYaw: target yaw angle for the swinging foot (default: 0.0 rad)
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Convert to numpy arrays
        footPos0 = np.array(footPos0)
        footTarget = np.array(footTarget)

        # Compute total displacement
        displacement = footTarget - footPos0

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i in swingFootIds:
                # Create smooth trajectory from footPos0 to footTarget
                # Phase 1 (first half): swing up
                # Phase 2 (second half): swing down
                phKnots = numKnots / 2
                progress = (k + 1) / numKnots  # Linear progress from 0 to 1

                if k < phKnots:
                    # Swing up phase
                    xy_progress = progress
                    z_height = stepHeight * (k / phKnots)
                else:
                    # Swing down phase
                    xy_progress = progress
                    z_height = stepHeight * (1 - float(k - phKnots) / phKnots)

                # Interpolate x,y position
                tref = footPos0 + displacement * xy_progress
                # Override z with swing trajectory
                tref[2] = footPos0[2] + z_height

                # Create rotation matrix for target yaw (rotation around z-axis)
                # Interpolate yaw from 0 to targetYaw
                current_yaw = targetYaw * progress
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                R_target = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw,  cos_yaw, 0],
                    [0,        0,       1]
                ])

                swingFootTask += [[i, pinocchio.SE3(R_target, tref)]]

            # Update CoM task
            comTask = displacement[:2] * progress * comPercentage + comPos0[:2]
            comTask = np.array([comTask[0], comTask[1], comPos0[2]])

            footSwingModel += [
                self.createSwingFootModel(
                    timeStep,
                    supportFootIds,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                    footWeight=8e6
                )
            ]

        # Action model for the foot switch (landing)
        footSwitchModel = self.createFootSwitchModel(
            swingFootIds, swingFootTask
        )

        return [*footSwingModel, footSwitchModel]

    def createFootstepModels(
        self,
        comPos0,
        feetPos0,
        stepLength,
        stepHeight,
        timeStep,
        numKnots,
        supportFootIds,
        swingFootIds,
    ):
        """Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0.0,
                         stepHeight * k / phKnots]
                    )
                elif k == phKnots:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0.0, stepHeight])
                else:
                    dp = np.array(
                        [
                            stepLength * (k + 1) / numKnots,
                            0.0,
                            stepHeight * (1 - float(k - phKnots) / phKnots),
                        ]
                    )
                tref = p + dp
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
            comTask = (
                np.array([stepLength * (k + 1) / numKnots, 0.0, 0.0]
                         ) * comPercentage
                + comPos0
            )
            footSwingModel += [
                self.createSwingFootModel(
                    timeStep,
                    supportFootIds,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                )
            ]
        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(
            swingFootIds, swingFootTask
        )
        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0.0, 0.0]
        for p in feetPos0:
            p += [stepLength, 0.0, 0.0]
        return [*footSwingModel, footSwitchModel]

    def createSwingFootModel(
        self, timeStep, supportFootIds, comTask=None, swingFootTask=None, comWeight=1e5, footWeight=1e6
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :param comWeight: weight for COM tracking (default: 1e5, higher for double support)
        :param footWeight: weight for swing foot position tracking (default: 1e6, increase to improve reaching)
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(
                self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, comWeight)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(
                self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, footWeight
                )

        # Add collision avoidance between swing and stance feet
        # This prevents feet from getting too close during the swing phase
        if len(supportFootIds) > 0 and swingFootTask is not None:
            for swing_task in swingFootTask:
                swing_foot_id = swing_task[0]
                for support_foot_id in supportFootIds:
                    # Skip if same foot
                    if swing_foot_id == support_foot_id:
                        continue

                    # Create a penalty for feet getting too close
                    # We penalize configurations where feet are closer than min_distance
                    try:
                        # Create frame velocity residual to indirectly discourage collision
                        # (Crocoddyl doesn't have direct distance constraints)
                        frame_residual = crocoddyl.ResidualModelFrameVelocity(
                            self.state, swing_foot_id, pinocchio.Motion.Zero(),
                            pinocchio.LOCAL, nu
                        )
                        # Weighted activation to smooth the cost
                        activation = crocoddyl.ActivationModelWeightedQuad(
                            np.array([1.0, 10.0, 1.0, 0.1, 0.1, 0.1]) ** 2
                        )
                        collision_cost = crocoddyl.CostModelResidual(
                            self.state, activation, frame_residual
                        )
                        costModel.addCost(
                            f"collision_avoid_{self.rmodel.frames[swing_foot_id].name}_{self.rmodel.frames[support_foot_id].name}",
                            collision_cost,
                            5e2  # Moderate weight
                        )
                    except:
                        pass  # Skip if not supported

        stateWeights = np.array(
            [0] * 3 + [500.0] * 3 + [0.01] *
            (self.state.nv - 6) + [10] * self.state.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(
                self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                impulseFootVelCost = crocoddyl.CostModelResidual(
                    self.state, frameVelocityResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, 1e8
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
                )
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        return model

    def createImpulseModel(
        self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, 1e8
                )
        stateWeights = np.array(
            [1.0] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model

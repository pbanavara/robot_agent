import mujoco
import numpy as np
import time

# ✅ Load MuJoCo model
MODEL_XML_PATH = "../franka_emika_panda/mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)


import numpy as np
import mujoco


def jacobian_ik(target_pos, max_iters=20, alpha=0.1, tol=1e-3):
    """
    Computes joint angles using Jacobian-based differential IK.
    """
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
    )  # Adjust site name if needed

    if site_id == -1:
        raise ValueError(
            "❌ ERROR: 'gripper' site not found in MuJoCo model. Check XML!"
        )

    for i in range(max_iters):
        mujoco.mj_forward(model, data)  # Update MuJoCo state

        # ✅ Get current end-effector position
        current_pos = np.copy(data.site_xpos[site_id])
        delta_pos = (
            np.array(target_pos) - current_pos
        )  # Difference between target and current position

        # ✅ Check convergence
        if np.linalg.norm(delta_pos) < tol:
            print(f"✅ Converged in {i+1} iterations!")
            break

        # ✅ Compute Jacobian (Position Only)
        jacobian_full = np.zeros((6, model.nv))  # 6D Jacobian (position + orientation)
        mujoco.mj_jacSite(
            model, data, jacobian_full[:3], jacobian_full[3:], site_id
        )  # Extract both pos & rot Jacobian

        jacobian_pos = jacobian_full[:3, : model.nu]  # Take only 3xN position Jacobian

        # ✅ Compute change in joint angles (dθ = J⁺ * Δx)
        dq = (
            np.linalg.pinv(jacobian_pos) @ delta_pos
        )  # Pseudo-inverse to solve for angles

        # ✅ Apply small updates to joint angles
        data.qpos[: model.nu] += alpha * dq
        mujoco.mj_forward(model, data)

        print(f"🔹 Iteration {i+1}: qpos = {data.qpos[:model.nu]}")

    return np.copy(data.qpos[: model.nu])  # Return final joint angles


# ✅ Function: Compute Joint Angles Using IK
def compute_joint_angles(target_position, max_ik_iterations=10):
    """
    Computes joint angles using MuJoCo's inverse kinematics solver.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
   
    if site_id == -1:
        raise ValueError(
            "❌ ERROR: 'end_effector' site not found in the MuJoCo model. Check the XML file!"
        )
    data.site_xpos[site_id] = np.array(target_position)


    before_ik = np.copy(data.qpos[:6])  # Save before calling IK

    mujoco.mj_inverse(model, data)  # Solve IK
    mujoco.mj_forward(model, data)  # Ensure MuJoCo state updates
    after_ik = np.copy(data.qpos[:6])  # Save after calling IK

    print(f"🔹 Before IK: {before_ik}")
    print(f"🔹 After IK: {after_ik}")

    if np.allclose(before_ik, after_ik):
        print("❌ `mj_inverse()` is not modifying qpos!")
    else:
        print("✅ `mj_inverse()` successfully updated qpos!")

    # ✅ Apply small perturbation to help solver escape local minima
    data.qpos[:6] += 0.005 * np.random.randn(6)  # Only modify first 6 joints

    # ✅ Run Forward Kinematics to update state
    mujoco.mj_forward(model, data)

    # ✅ Print joint angles before IK
    print(f"🔹 Before IK: qpos = {data.qpos[:6]}")

    # ✅ Run multiple iterations of IK
    for i in range(max_ik_iterations):
        mujoco.mj_inverse(model, data)
        mujoco.mj_forward(model, data)  # Update state after inverse kinematics
        print(f"🔹 Iteration {i+1}: qpos = {data.qpos[:6]}")

    print(f"🔹 After IK: qpos = {data.qpos[:6]}")

    # Get the computed joint angles
    return np.copy(data.qpos[: model.nu])


def move_arm_to_target(target_position, threshold=0.01, max_iterations=50):
    """
    Directly solves for joint angles to move the end-effector to the target.
    """
    # Compute joint angles using MuJoCo's IK solver
    joint_angles = jacobian_ik(target_position)

    # ✅ Directly set the joint angles
    data.qpos[:6] = joint_angles[:6]  # Apply only the first 6 joints
    mujoco.mj_forward(model, data)  # Update MuJoCo state

    # ✅ Get current end-effector position
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    current_position = np.copy(data.site_xpos[site_id])

    # ✅ Compute delta (error)
    delta = np.linalg.norm(np.array(target_position) - current_position)

    print(f"Target Pos: {target_position}, Reached Pos: {current_position}, Delta: {delta}")

    return {"status": "success" if delta < 0.01 else "failed", "delta": delta}



def move_single_joint(joint_index, step_size=0.05, max_iterations=20):
    """
    Moves a single joint gradually to debug motion.
    """
    for i in range(max_iterations):
        data.qpos[joint_index] += step_size  # Move only the specified joint
        mujoco.mj_forward(model, data)  # Update MuJoCo state

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        current_position = np.copy(data.site_xpos[site_id])

        print(
            f"Iteration {i}: Joint {joint_index} moved, New End-Effector Pos: {current_position}"
        )

        time.sleep(0.1)  # Slow down for visualization


# ✅ Launch MuJoCo Viewer and Execute Movement
with mujoco.viewer.launch_passive(model, data) as viewer:
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
    ) 
    target_position = [0.9, 0.9, 0.9]  # Example target position
    for i in range(model.nsite):
        print(f"Site {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")

    print("🔍 Checking Reachability...")
    print("Target Position:", target_position)
    print("Current End-Effector Position:", data.site_xpos[site_id])

    # Compute max possible reach (sum of all link lengths)
    link_lengths = [0.333, 0.316, 0.0825, 0.384, 0.107]  # From your Panda XML
    max_reach = sum(link_lengths)
    target_distance = np.linalg.norm(np.array(target_position))

    print(f"🦾 Max Reach: {max_reach}, Target Distance: {target_distance}")
    if target_distance > max_reach:
        print("❌ Target is out of reach!")
    
    
    # ✅ Move the arm while updating the viewer in real-time
    while viewer.is_running():
        #result = move_arm_to_target(target_position)
        # ✅ Step the simulation continuously for smooth movement
        result = move_arm_to_target(target_position)
        if result["status"] == "success":
            print("✅ Target reached!")
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(2)
    
    
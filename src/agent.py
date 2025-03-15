import openai
from openai import OpenAI
import numpy as np
import mujoco
import os
import time

openai.api_key = os.environ.get("OPENAI_API_KEY")

import openai
import numpy as np

openai.api_key = "YOUR_OPENAI_API_KEY"

def get_end_effector_position(model, data):
    """
    Retrieves the current position of the end-effector.
    """
    # Get the site ID of the end-effector
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    if site_id == -1:
        raise ValueError("❌ ERROR: End-effector site 'gripper' not found in the MuJoCo model!")

    # Extract position
    position = np.copy(data.site_xpos[site_id])  # (x, y, z) coordinates

    print(f"🔍 End-Effector Position: {position}")  # Debugging
    return position

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

        #print(f"🔹 Iteration {i+1}: qpos = {data.qpos[:model.nu]}")

    return np.copy(data.qpos[: model.nu])

def get_adjusted_target(current_position, 
                        target_position, delta, max_attempts=3):
    """
    Adjusts the target position based on the current position and delta.
    """
    client = OpenAI()
    for _ in range(max_attempts):
        if delta < 0.05:
            print("✅ Target reached!")
            return target_position  # No adjustment needed

        # Format LLM prompt
        prompt = f"""
        The robotic arm is trying to reach {target_position} but is currently at {current_position}.
        The difference (delta) is {delta}. Suggest a new target position to improve reachability.
        The new target should be within the robot's workspace and avoid unnecessary movement.
        Return only the new position as a list of [x, y, z] values.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        try:
            new_target = eval(response.choices[0].message.content)
            if isinstance(new_target, list) and len(new_target) == 3:
                print(f"🔄 LLM adjusted target to: {new_target}")
                return new_target
        except Exception as e:
            print(f"⚠️ Error in LLM output: {e}")

    print("❌ LLM failed to suggest a valid target. Keeping original target.")
    return target_position  # Fallback to the original target

def smooth_joint_movement(model, data, target_qpos, steps=50):
    """
    Smoothly interpolates from the current joint angles to the target joint angles over multiple steps.
    """
    current_qpos = np.copy(data.qpos[: model.nu])  # Current joint angles

    for i in range(steps):
        alpha = (i + 1) / steps  # Linearly interpolate between 0 and 1
        data.qpos[: model.nu] = (
            1 - alpha
        ) * current_qpos + alpha * target_qpos  # Smooth transition

        mujoco.mj_step(model, data)  # Step the simulation
        viewer.sync()  # Update the MuJoCo viewer
        time.sleep(0.02)  # Small delay for visible motion

    return True

def move_arm_with_llm(target_position, model, data, max_iterations=10):
    """
    Moves the robot arm to the target, dynamically adjusting the target using LLM if needed.
    """
    current_position = get_end_effector_position(model, data)
    delta = np.linalg.norm(np.array(target_position) - np.array(current_position))

    for iteration in range(max_iterations):
        print(
            f"🔹 Iteration {iteration}: Current Pos {current_position}, Delta {delta}"
        )

        # If the target is too far, ask LLM for a better one
        if delta > 0.1:  # Example threshold for "too far"
            print("⚠️ Target may be unreachable! Asking LLM for adjustment...")
            target_position = get_adjusted_target(
                current_position, target_position, delta
            )

        # Compute joint angles (Using IK or RL)
        joint_angles = jacobian_ik(target_position)
        smooth_joint_movement(model, data, joint_angles, steps=50)

        # Apply the new angles
        #data.qpos[: len(joint_angles)] = joint_angles
        #mujoco.mj_forward(model, data)

        # Get updated position
        current_position = get_end_effector_position(model, data)
        delta = np.linalg.norm(np.array(target_position) - np.array(current_position))

        # Check if we have reached the goal
        if delta < 0.01:
            print("✅ Target reached successfully!")
            return {"status": "success", "position": current_position, "delta": delta}

    print("❌ Failed to reach target after max iterations")
    return {"status": "failed", "position": current_position}

if __name__ == "__main__":
    MODEL_XML_PATH = "../franka_emika_panda/mjx_panda.xml"
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    target_position = [0.8, 0.9, -0.9]

    data = mujoco.MjData(model)

    # Apply uniform scaling to all positions and sizes
    scale_factor = 0.8
    model.body_pos *= scale_factor
    model.geom_size *= scale_factor
    model.geom_pos *= scale_factor
    model.jnt_range *= scale_factor  # Optional: Scale joint limits

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 180  # Rotate view
        viewer.cam.elevation = -10  # Adjust height angle
        viewer.cam.distance = 1.5  # Zoom level
        viewer.cam.lookat[:] = [0, 0, 0.5]  # Focus on robot
        while viewer.is_running():
            result = move_arm_with_llm(target_position, model, data)
            if result["status"] == "success":
                print("✅ Task Complete!")
            else:
                print("🔄 Adjusting target and retrying...")
                target_position = get_adjusted_target(result["position"], target_position, 0.05)
            viewer.sync()
            time.sleep(10)
import openai
import json
import os
import nearai
from openai import OpenAI
import numpy as np
import mujoco
import os
import time
import numpy as np
from nearai.agents.environment import Environment


os.environ["MUJOCO_GL"] = "egl"
hub_url = "https://api.near.ai/v1"

# Login to NEAR AI Hub using nearai CLI.
# Read the auth object from ~/.nearai/config.json

# GLOBALS
MODEL_XML_PATH = "../franka_emika_panda/mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
target_position = [0.8, 0.9, -0.9]
data = mujoco.MjData(model)
auth = nearai.config.load_config_file()["auth"]
signature = json.dumps(auth)
client = openai.OpenAI(base_url=hub_url, api_key=signature)

# ✅ Create OpenGL context
gl_context = mujoco.GLContext(1280, 720)  # Offscreen buffer
gl_context.make_current()  # Activate EGL context

# ✅ Create MuJoCo rendering context
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# ✅ Define camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.azimuth = -180  # Rotate view
cam.elevation = -20  # Adjust height
cam.distance = 2.5  # Zoom level
cam.lookat[:] = [0, 0, 0]  # Focus on robot

# ✅ Set up MuJoCo visualization
scene = mujoco.MjvScene(model, maxgeom=1000)
option = mujoco.MjvOption()
mujoco.mjv_defaultOption(option)

# ✅ Define framebuffer (to prevent OpenGL errors)
viewport = mujoco.MjrRect(0, 0, 800, 600)  # 800x600 resolution
#mujoco.mjr_makeContext(model, context, mujoco.mjtFontScale.mjFONTSCALE_150)

def get_end_effector_position(model, data):
    """
    Retrieves the current position of the end-effector.
    """
    # Get the site ID of the end-effector
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    if site_id == -1:
        raise ValueError(
            "❌ ERROR: End-effector site 'gripper' not found in the MuJoCo model!"
        )

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

        # print(f"🔹 Iteration {i+1}: qpos = {data.qpos[:model.nu]}")

    return np.copy(data.qpos[: model.nu])


def get_adjusted_target(current_position, 
                        target_position, 
                        delta,
                        client,
                        max_attempts=3):
    """
    Adjusts the target position based on the current position and delta.
    """
    for _ in range(max_attempts):
        if delta < 0.05:
            print("✅ Target reached!")
            return target_position  # No adjustment needed

        # Format LLM prompt
        prompt = f"""
        The robotic arm is trying to reach {target_position} but is currently at {current_position}.
        The difference (delta) is {delta}. Suggest a new target position to improve reachability.
        The new target should be within the robot's workspace and avoid unnecessary movement.
        Do not return any other text or values or explanations.
        Return only the new position as a list of [x, y, z] values.
        """

        response = client.chat.completions.create(
            model="fireworks::accounts/fireworks/models/qwen2p5-72b-instruct",
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
        #viewer.sync()  # Update the MuJoCo viewer
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
                                    current_position, 
                                    target_position, 
                                    delta, 
                                    client
            )

        # Compute joint angles (Using IK or RL)
        joint_angles = jacobian_ik(target_position)
        smooth_joint_movement(model, data, joint_angles, steps=50)
        save_screenshot(model, data, filename=f"screenshot_{int(time.time())}.png")

        # Apply the new angles
        # data.qpos[: len(joint_angles)] = joint_angles
        # mujoco.mj_forward(model, data)

        # Get updated position
        current_position = get_end_effector_position(model, data)
        delta = np.linalg.norm(np.array(target_position) - np.array(current_position))

        # Check if we have reached the goal
        if delta < 0.01:
            print("✅ Target reached successfully!")
            return {"status": "success", "position": current_position, "delta": delta}

    print("❌ Failed to reach target after max iterations")
    return {"status": "failed", "position": current_position}

import mujoco
import numpy as np
import imageio


def save_screenshot(model, data, filename="screenshot.png"):
    """Captures a screenshot from the MuJoCo viewer and saves it."""
    rgb_buffer = np.zeros((600, 800, 3), dtype=np.uint8)  # Make sure shape matches viewport
    depth_buffer = np.zeros((600, 800, 1), dtype=np.float32)  # Optional: Depth buffer

    # Update scene
    mujoco.mjv_updateScene(model, data, option, None, cam, 
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    # ✅ Bind framebuffer and render
    mujoco.mjr_render(viewport, scene, context)

    # ✅ Read pixels correctly
    mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewport, context)

    # ✅ Save image
    imageio.imwrite(filename, rgb_buffer)
    print(f"📸 Screenshot saved: {filename}")


if __name__ == "__main__":
   
    while True:
        result = move_arm_with_llm(target_position, model, data)  # Adjust as per your function

        if result["status"] == "success":
            print("✅ Task Complete!")
            save_screenshot(model, data, filename=f"screenshot_{int(time.time())}.png")  # Save screenshot
            user_input = input("Do you want to reach a new target? (y/n): ")
            if user_input.lower() != "y":
                break
        else:
            print("🔄 Adjusting target and retrying...")
            target_position = get_adjusted_target(result["position"], target_position, 0.05, client)

        mujoco.mj_step(model, data)  # Step simulation
        save_screenshot(model, data, 
                        filename=f"screenshot_{int(time.time())}.png")
        time.sleep(2)

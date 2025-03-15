#### This is a submission for the nearAI hackathon.

This has to be run locally since it uses a Mujoco viewer for robot simulation.
There is an online headless script but that needs Mujoco libraries as well.

I will submit a PR to include those libraries for NEAR.

#### To run the code
python nearai_agent.py ( for the viewer)

python nearai_agent_headless.py ( for the headless version)

#### Benchmark
please see src/logs.txt which includes the given target and how the robot is
reaching that target powered by the LLM instructions.

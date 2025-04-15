# Model-Free Reinforcement Learning Robust To Irreversible Events in Robotic Manipulation

## Project Overview

This project is part of a Master’s thesis focused on the **design of a model-free reinforcement learning algorithm robust to irreversible events** in robotic manipulation. The algorithm is applied to **bimanual re-orientation and lift tasks** using two **Franka Emika Panda** robots, each equipped with a **custom semispherical gripper**.

The project builds on the [**robosuite** framework](https://github.com/ARISE-Initiative/robosuite), which is used for simulating complex robotic manipulation environments.

---

## Installation

The project uses a Conda environment, named robosuite-test, with Robosuite installed according to the official documentation:

👉 [Robosuite Installation Guide](https://robosuite.ai/docs/installation.html)

---

## Environments

Two custom environments were developed specifically for the manipulation tasks in this study:

- **Milk Carton Reorientation Task**  
  File: `robosuite/project/reorientation/two_arm_milk_lift.py`

- **Can Lift Task**  
  File: `robosuite/project/lift/two_arm_can_lift.py`

These environments simulate dual-arm manipulation with task-specific dynamics and constraints and robot initialization parameters.

---

## Custom Grippers

A pair of **semispherical grippers** were designed and mounted on the Franka Emika robots to enable robust contact and manipulation.  
You can find the gripper model here:

- `robosuite/models/assets/grippers/spherical_gripper.xml`

---

## Usage

### Run the Can Lift Task

```bash
~/anaconda3/envs/robosuite-test/bin/python ~/robosuite/robosuite/project/lift/main_can.py Can
```

### Run the Milk Reorientation Task
```bash
~/anaconda3/envs/robosuite-test/bin/python ~/robosuite/robosuite/project/reorientation/main_osc_milk.py Milk
```

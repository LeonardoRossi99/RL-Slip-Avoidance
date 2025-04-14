def set_environment(environment_configs):

    if environment_configs=="Cereal":
        env_name_= "TwoArmCerealLift"
        robot0_="Panda1"
        robot1_="Panda2"
        controllers_type= "JOINT_TORQUE"
        grippers_type= "SphericalGripper"
        arm_configuration= "single-arm-parallel"
        obj_position= "cereal_pos"
        torque_sensor_name= "gripper1_contact_torque"
        force_sensor_name= "gripper1_contact_force"
        dist_sensor_eef= 0.06
        robot0_eef_quat= "robot0_eef_quat"
        robot1_eef_quat= "robot1_eef_quat"
        obj_quat = "cereal_quat"


    if environment_configs=="Lemon":
        env_name_= "TwoArmLemonLift"
        robot0_="Panda0Lemon"
        robot1_="Panda1Lemon"
        controllers_type= "JOINT_TORQUE"
        grippers_type= "SphericalGripper"
        arm_configuration= "single-arm-parallel"
        obj_position= "lemon_pos"
        torque_sensor_name= "gripper1_contact_torque"
        force_sensor_name= "gripper1_contact_force"
        dist_sensor_eef= 0.06
        robot0_eef_quat= "robot0_eef_quat"
        robot1_eef_quat= "robot1_eef_quat"
        obj_quat = "lemon_quat"



    if environment_configs=="Milk":
        env_name_= "TwoArmMilkLift"
        robot0_="Panda0Milk" 
        robot1_= "Panda1Milk"
        controllers_type= "JOINT_TORQUE"
        grippers_type= "SphericalGripper"
        arm_configuration= "single-arm-parallel"
        obj_position= "milk_pos"
        torque_sensor_name= "gripper1_contact_torque"
        force_sensor_name= "gripper1_contact_force"
        dist_sensor_eef= 0.06
        robot0_eef_quat= "robot0_eef_quat"
        robot1_eef_quat= "robot1_eef_quat"
        obj_quat = "milk_quat"



    if environment_configs=="Can":
        env_name_= "TwoArmCanLift"
        robot0_= "Panda0Can"
        robot1_= "Panda1Can"
        controllers_type= "JOINT_TORQUE"
        grippers_type= "SphericalGripper"
        arm_configuration= "single-arm-parallel"
        obj_position= "can_pos"
        torque_sensor_name= "gripper1_contact_torque"
        force_sensor_name= "gripper1_contact_force"
        dist_sensor_eef= 0.06
        robot0_eef_quat= "robot0_eef_quat"
        robot1_eef_quat= "robot1_eef_quat"
        obj_quat = "can_quat"

    if environment_configs=="Box":
        env_name_= "TwoArmCubeLift"
        robot0_= "Panda0Can"
        robot1_= "Panda1Can"
        controllers_type= "JOINT_TORQUE"
        grippers_type= "SphericalGripper"
        arm_configuration= "single-arm-parallel"
        obj_position= "cube_pos"
        torque_sensor_name= "gripper1_contact_torque"
        force_sensor_name= "gripper1_contact_force"
        dist_sensor_eef= 0.06
        robot0_eef_quat= "robot0_eef_quat"
        robot1_eef_quat= "robot1_eef_quat"
        obj_quat = "cube_quat"
        




    return env_name_, robot0_, robot1_, controllers_type, grippers_type, arm_configuration, obj_position, torque_sensor_name, force_sensor_name, dist_sensor_eef, robot0_eef_quat, robot1_eef_quat, obj_quat
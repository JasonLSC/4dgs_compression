import numpy as np
import json
import copy

def rotation_matrix_to_quaternion(R):
    # 确保输入是一个有效的旋转矩阵
    assert R.shape == (3, 3), "输入必须是一个3x3的旋转矩阵"
    
    # 计算四元数的各个分量
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def quaternion_to_rotation_matrix(q):
    # 确保输入是一个有效的四元数
    assert len(q) == 4, "输入必须是一个四元数，包含四个元素"
    
    # 提取四元数的各个分量
    qw, qx, qy, qz = q
    
    # 计算旋转矩阵的各个元素
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R

def slerp(q0, q1, t):
    """Spherical linear interpolation (SLERP) between two quaternions."""
    dot = np.dot(q0, q1)
    
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result)
        return result
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q0) + (s1 * q1)

def interpolate_camera_poses(qvec0, tvec0, qvec1, tvec1, n_frames):
    # qvec0 = np.array(qvec0)
    # qvec1 = np.array(qvec1)
    # tvec0 = np.array(tvec0)
    # tvec1 = np.array(tvec1)
    
    q_interpolated = []
    t_interpolated = []
    
    for i in range(n_frames):
        t = i / (n_frames - 1)
        q_interp = slerp(qvec0, qvec1, t)
        t_interp = (1 - t) * tvec0 + t * tvec1
        
        q_interpolated.append(q_interp)
        t_interpolated.append(t_interp)
    
    return np.array(q_interpolated), np.array(t_interpolated)

if __name__ == "__main__":
    scene_path = "/work/Users/lisicheng/Dataset/INVR_n3d_like/Pierrot/"
    scene_name = "Pierrot"
    frame_num = 96

    with open("/work/Users/lisicheng/Dataset/INVR_n3d_like/Pierrot/transforms_test.json", "r") as fp:
        input_dict = json.load(fp)
        cam_poses = input_dict["frames"]

    pose_trace = copy.deepcopy(input_dict)
    pose_trace["frames"] = []
    
    pose0 = np.array(cam_poses[0]["transform_matrix"])
    time0 = cam_poses[0]["time"]
    pose1 = np.array(cam_poses[-1]["transform_matrix"])
    time1 = cam_poses[-1]["time"]

    qvec0 = rotation_matrix_to_quaternion(pose0[:3,:3])
    tvec0 = pose0[:3,-1].transpose()
    qvec1 = rotation_matrix_to_quaternion(pose1[:3,:3])
    tvec1 = pose1[:3,-1].transpose()

    q_interpolated, t_interpolated = interpolate_camera_poses(qvec0, tvec0, qvec1, tvec1, frame_num)

    for i in range(frame_num):
        pose = np.eye(4)
        rot_mat = quaternion_to_rotation_matrix(q_interpolated[i])
        pos = t_interpolated[i].transpose()

        pose[:3, :3] = rot_mat
        pose[:3, -1] = pos
        
        pose_trace["frames"].append({
            "transform_matrix": pose.tolist(),
            "time": i/30,
        })
    
    with open("/work/Users/lisicheng/Dataset/INVR_n3d_like/Pierrot/pose_trace.json", "w") as fp:
        json.dump(pose_trace,fp,indent=True)
    # check
    # print(np.isclose(pose0[:3,:3],quaternion_to_rotation_matrix(qvec0)))

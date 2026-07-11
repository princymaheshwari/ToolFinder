import numpy as np

# ── Hardware offsets (all in mm, set to match your real system) ──────────────

# Plank origin in world space
PLANK_X = 0.0
PLANK_Y = 0.0
PLANK_Z = 200.0

# Yaw servo axis position relative to plank origin
YAW_FROM_PLANK_X = 0.0
YAW_FROM_PLANK_Y = 0.0
YAW_FROM_PLANK_Z = 0.0

# Yaw axle vector in yaw servo local frame (rotates with yaw)
YAW_AXLE_DX = 50.0
YAW_AXLE_DY = 0.0
YAW_AXLE_DZ = 0.0

# Pitch servo axis offset from end of yaw axle (in yaw-rotated frame)
PITCH_FROM_YAW_END_X = 0.0
PITCH_FROM_YAW_END_Y = 0.0
PITCH_FROM_YAW_END_Z = -20.0

# Pitch axle vector in pitch servo local frame (default pointing down)
PITCH_AXLE_DX = 0.0
PITCH_AXLE_DY = 0.0
PITCH_AXLE_DZ = -30.0

# Torch offset from end of pitch axle (in pitch-rotated frame)
TORCH_FROM_PITCH_END_X = 0.0
TORCH_FROM_PITCH_END_Y = 0.0
TORCH_FROM_PITCH_END_Z = -10.0


# ── Rotation helpers ─────────────────────────────────────────────────────────

def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])

def rot_rodrigues(axis, deg):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ]
    ])


# ── Forward kinematics ────────────────────────────────────────────────────────

def get_torch_pos_and_direction(yaw_deg, pitch_deg):
    """
    Walk the full kinematic chain.
    Returns (torch_origin, torch_direction) both in world space.
    torch_direction is the unit vector the torch is pointing.
    Default torch points along -Z, tilted by pitch around yaw arm's local X.
    """
    Ry = rot_z(yaw_deg)

    # 1. Plank origin
    plank = np.array([PLANK_X, PLANK_Y, PLANK_Z])

    # 2. Yaw servo axis
    yaw_axis = plank + np.array([YAW_FROM_PLANK_X,
                                  YAW_FROM_PLANK_Y,
                                  YAW_FROM_PLANK_Z])

    # 3. End of yaw axle (rotates with yaw)
    yaw_axle_end = yaw_axis + Ry @ np.array([YAW_AXLE_DX,
                                              YAW_AXLE_DY,
                                              YAW_AXLE_DZ])

    # 4. Pitch servo axis (offset from yaw axle end, rotates with yaw)
    pitch_axis = yaw_axle_end + Ry @ np.array([PITCH_FROM_YAW_END_X,
                                                PITCH_FROM_YAW_END_Y,
                                                PITCH_FROM_YAW_END_Z])

    # 5. Pitch rotation is around the yaw arm's local X axis
    pitch_rot_axis = Ry @ np.array([1.0, 0.0, 0.0])
    Rp = rot_rodrigues(pitch_rot_axis, pitch_deg)

    # 6. End of pitch axle (through both rotations)
    pitch_axle_end = pitch_axis + Rp @ (Ry @ np.array([PITCH_AXLE_DX,
                                                         PITCH_AXLE_DY,
                                                         PITCH_AXLE_DZ]))

    # 7. Torch start position (through both rotations)
    torch_pos = pitch_axle_end + Rp @ (Ry @ np.array([TORCH_FROM_PITCH_END_X,
                                                        TORCH_FROM_PITCH_END_Y,
                                                        TORCH_FROM_PITCH_END_Z]))

    # 8. Torch pointing direction — default is -Z, rotated through pitch (and yaw)
    default_dir      = np.array([0.0, 0.0, -1.0])
    torch_direction  = Rp @ (Ry @ default_dir)
    torch_direction /= np.linalg.norm(torch_direction)

    return torch_pos, torch_direction


def get_torch_ground_hit(yaw_deg, pitch_deg):
    """
    Cast a ray from torch origin along torch direction.
    Returns XY world point where it hits Z=0, or None if it never does.
    """
    origin, direc = get_torch_pos_and_direction(yaw_deg, pitch_deg)

    if abs(direc[2]) < 1e-6:
        return None
    t = -origin[2] / direc[2]
    if t < 0:
        return None

    hit = origin + direc * t
    return np.array([hit[0], hit[1]])


# ── Solvers ───────────────────────────────────────────────────────────────────

def solve_yaw(poi_x, poi_y):
    """
    Sweep yaw -90..+90.
    Minimise distance from green line (perpendicular to arm, through yaw axis) to POI.
    """
    best_angle = None
    best_dist  = float('inf')

    plank  = np.array([PLANK_X, PLANK_Y, PLANK_Z])
    yaw_axis_world = plank + np.array([YAW_FROM_PLANK_X,
                                        YAW_FROM_PLANK_Y,
                                        YAW_FROM_PLANK_Z])
    anchor = np.array([yaw_axis_world[0], yaw_axis_world[1], 0.0])

    for step in range(-1000, 1000, 1):
        deg = step * 90.0 / 1000.0

        Ry      = rot_z(deg)
        arm_dir = Ry @ np.array([1.0, 0.0, 0.0])
        perp    = np.array([-arm_dir[1], arm_dir[0], 0.0])
        perp   /= np.linalg.norm(perp)

        hp   = np.array([poi_x - anchor[0], poi_y - anchor[1], 0.0])
        dist = np.linalg.norm(hp - np.dot(hp, perp) * perp)

        if dist < best_dist:
            best_dist  = dist
            best_angle = deg

    return best_angle


def solve_pitch(poi_x, poi_y, yaw_deg):
    """
    Sweep pitch -90..+90.
    Minimise distance between torch beam ground hit and POI.
    """
    best_angle = None
    best_dist  = float('inf')

    for step in range(-1000, 1000, 1):
        pitch_deg = step * 90.0 / 1000.0

        ground_xy = get_torch_ground_hit(yaw_deg, pitch_deg)
        if ground_xy is None:
            continue

        dist = np.linalg.norm(ground_xy - np.array([poi_x, poi_y]))

        if dist < best_dist:
            best_dist  = dist
            best_angle = pitch_deg

    return best_angle


def solve(poi_x, poi_y):
    """
    Returns (yaw_deg, pitch_deg) to aim torch beam at (poi_x, poi_y, 0).
    """
    yaw_deg   = solve_yaw(poi_x, poi_y)
    pitch_deg = solve_pitch(poi_x, poi_y, yaw_deg)
    return yaw_deg, pitch_deg


# ── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    poi = (80.0, 40.0)

    yaw, pitch    = solve(*poi)
    ground_xy     = get_torch_ground_hit(yaw, pitch)
    torch_pos, torch_dir = get_torch_pos_and_direction(yaw, pitch)
    err           = np.linalg.norm(ground_xy - np.array(poi)) if ground_xy is not None else float('inf')

    print(f"POI              : {poi}")
    print(f"Yaw angle        : {yaw:.4f}°")
    print(f"Pitch angle      : {pitch:.4f}°")
    print(f"Torch origin     : ({torch_pos[0]:.3f}, {torch_pos[1]:.3f}, {torch_pos[2]:.3f})")
    print(f"Torch direction  : ({torch_dir[0]:.3f}, {torch_dir[1]:.3f}, {torch_dir[2]:.3f})")
    print(f"Ground hit       : {ground_xy}")
    print(f"Error            : {err:.6f} mm")
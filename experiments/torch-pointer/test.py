import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# --- Config ---
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FOV       = 60.0
NEAR_CLIP   = 0.1
FAR_CLIP    = 10000.0
SCROLL_SPEED  = 2.0

# --- Frame dimensions ---
WIDTH  = 100.0
HEIGHT = 60.0

# --- Camera State ---
cam_pos   = np.array([0.0, -80.0, 40.0])
cam_yaw   = 0.0
cam_pitch = -20.0

# --- POI State ---
poi_pos  = np.array([0.0, 0.0, 0.0])
POI_STEP = 0.03

# ── Plank ────────────────────────────────────────────────────────────────────
PLANK_POS      = np.array([0.0, 0.0, 20.0])   # centre of plank
PLANK_W      = 20.0
PLANK_D      = 5.0
PLANK_H      = 2.5

# ── Yaw servo geometry ───────────────────────────────────────────────────────
# Axle sits at the +X end of the plank, below it
YAW_AXLE_X_OFFSET  =  PLANK_W / 2.0   # flush with +X end of plank
YAW_AXLE_Z_OFFSET  = -(PLANK_H / 2.0 + 1.2)  # hang below plank by this gap
YAW_AXLE_LEN     =  8.0
YAW_AXLE_RADIUS  =  1.0

# ── Arm from yaw axle tip to pitch servo ─────────────────────────────────────
YAW_ARM_RADIAL   =  YAW_AXLE_LEN / 2.0   # offset along axle (+X tip)
YAW_ARM_Z_DROP   = -2.5           # pitch servo hangs this far below axle

# ── Pitch servo geometry ──────────────────────────────────────────────────────
PITCH_AXLE_LEN   =  6.0
PITCH_AXLE_RADIUS  =  0.8
PITCH_BODY_SIZE  =  2.5

# ── Servo motion ─────────────────────────────────────────────────────────────
ANGLE_RATE     = 60.0   # degrees per second while key held

# --- Servo angles ---
yServoAngle = 0.0
pServoAngle = 0.0

# --- Mouse State ---
last_mouse = None
left_down  = False
right_down = False


# ------------------------------------------------------------------ camera --

def get_forward():
  yaw_r   = np.radians(cam_yaw)
  pitch_r = np.radians(cam_pitch)
  return np.array([
    np.sin(yaw_r) * np.cos(pitch_r),
    np.cos(yaw_r) * np.cos(pitch_r),
    np.sin(pitch_r)
  ])

def get_right():
  forward  = get_forward()
  world_up = np.array([0.0, 0.0, 1.0])
  right  = np.cross(forward, world_up)
  n    = np.linalg.norm(right)
  return right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

def get_up():
  return np.cross(get_right(), get_forward())

def apply_camera():
  forward = get_forward()
  look  = cam_pos + forward
  up    = get_up()
  glLoadIdentity()
  gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
        look[0],  look[1],  look[2],
        up[0],    up[1],    up[2])

def pixel_to_world_translation(dx, dy):
  half_h     = np.tan(np.radians(FOV / 2.0))
  px_per_unit  = WINDOW_HEIGHT / (2.0 * half_h)
  dist     = np.linalg.norm(cam_pos)
  world_per_px = dist / px_per_unit
  cam_pos[:]  -= get_right() * dx * world_per_px
  cam_pos[:]  += get_up()  * dy * world_per_px


# --------------------------------------------------------- servo kinematics --

def yaw_rotation_matrix():
  a = np.radians(yServoAngle)
  c, s = np.cos(a), np.sin(a)
  return np.array([[ c, -s, 0],
           [ s,  c, 0],
           [ 0,  0, 1]])

def get_yaw_axle_pos():
  """
  Yaw axle centre: at the +X end of the plank, hanging below it.
  """
  return np.array([
    PLANK_POS[0] + YAW_AXLE_X_OFFSET,
    PLANK_POS[1],
    PLANK_POS[2] + YAW_AXLE_Z_OFFSET
  ])

def get_pitch_axle_pos():
  """
  Pitch servo sits at the +X tip of the yaw axle, dropped down by YAW_ARM_Z_DROP.
  The radial offset rotates with yServoAngle.
  """
  R    = yaw_rotation_matrix()
  offset = R @ np.array([YAW_ARM_RADIAL, 0.0, 0.0])
  base   = get_yaw_axle_pos()
  return np.array([
    base[0] + offset[0],
    base[1] + offset[1],
    base[2] + YAW_ARM_Z_DROP
  ])

def get_barrel_direction():
  """
  Default barrel = -Z (pointing down).
  Pitch rotates it around the yaw arm's local X axis.
  """
  R     = yaw_rotation_matrix()
  local_x = R @ np.array([1.0, 0.0, 0.0])
  default = np.array([0.0, 0.0, -1.0])
  p     = np.radians(pServoAngle)
  c, s  = np.cos(p), np.sin(p)
  ax    = local_x
  barrel  = (default * c
         + np.cross(ax, default) * s
         + ax * np.dot(ax, default) * (1.0 - c))
  return barrel / np.linalg.norm(barrel)

def get_ground_hit():
  origin = get_pitch_axle_pos()
  direc  = get_barrel_direction()
  if abs(direc[2]) < 1e-6:
    return None
  t = -origin[2] / direc[2]
  if t < 0:
    return None
  return origin + direc * t


# ------------------------------------------------------------- draw helpers --

def _apply_pose(x, y, z, pitch_deg, yaw_deg):
  glTranslatef(x, y, z)
  glRotatef(yaw_deg,   0.0, 0.0, 1.0)
  glRotatef(pitch_deg, 1.0, 0.0, 0.0)

def draw_cuboid(x, y, z, pitch_deg, yaw_deg,
        width=10.0, depth=10.0, height=10.0,
        color=(0.2, 0.6, 1.0)):
  hw, hd, hh = width/2, depth/2, height/2
  verts = np.array([
    [-hw,-hd,-hh],[ hw,-hd,-hh],[ hw, hd,-hh],[-hw, hd,-hh],
    [-hw,-hd, hh],[ hw,-hd, hh],[ hw, hd, hh],[-hw, hd, hh],
  ])
  faces   = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]
  normals = [(0,0,-1),(0,0,1),(0,-1,0),(0,1,0),(1,0,0),(-1,0,0)]
  edges   = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
  glPushMatrix()
  _apply_pose(x, y, z, pitch_deg, yaw_deg)
  glColor3f(*color)
  glBegin(GL_QUADS)
  for face, normal in zip(faces, normals):
    glNormal3f(*normal)
    for vi in face: glVertex3fv(verts[vi])
  glEnd()
  glColor3f(1, 1, 1)
  glBegin(GL_LINES)
  for a, b in edges:
    glVertex3fv(verts[a]); glVertex3fv(verts[b])
  glEnd()
  glPopMatrix()

def draw_cylinder_along_axis(cx, cy, cz, axis, length, radius, color, segments=32):
  axis = np.array(axis, dtype=float)
  axis /= np.linalg.norm(axis)
  z   = np.array([0.0, 0.0, 1.0])
  dot   = np.dot(z, axis)
  cross = np.cross(z, axis)
  cn  = np.linalg.norm(cross)

  glPushMatrix()
  glTranslatef(cx, cy, cz)
  if cn > 1e-6:
    glRotatef(np.degrees(np.arctan2(cn, dot)), cross[0], cross[1], cross[2])
  elif dot < 0:
    glRotatef(180.0, 1.0, 0.0, 0.0)

  half   = length / 2.0
  angles = [i * 2 * np.pi / segments for i in range(segments)]
  circle = [(np.cos(a) * radius, np.sin(a) * radius) for a in angles]

  glColor3f(*color)
  glBegin(GL_QUAD_STRIP)
  for i in range(segments + 1):
    cx2, cy2 = circle[i % segments]
    glNormal3f(cx2/radius, cy2/radius, 0.0)
    glVertex3f(cx2, cy2, -half)
    glVertex3f(cx2, cy2,  half)
  glEnd()
  for cap_z, nz in [(-half, -1.0), (half, 1.0)]:
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0, 0, nz)
    glVertex3f(0, 0, cap_z)
    for i in range(segments + 1):
      cx2, cy2 = circle[i % segments]
      glVertex3f(cx2, cy2, cap_z)
    glEnd()
  glColor3f(0.6, 0.6, 0.6)
  for cap_z in [-half, half]:
    glBegin(GL_LINE_LOOP)
    for cx2, cy2 in circle:
      glVertex3f(cx2, cy2, cap_z)
    glEnd()
  glPopMatrix()

def draw_sphere(pos, radius=1.5, color=(1.0, 0.2, 0.2)):
  glPushMatrix()
  glTranslatef(pos[0], pos[1], 0.0)
  glColor3f(*color)
  quad = gluNewQuadric()
  gluSphere(quad, radius, 32, 32)
  gluDeleteQuadric(quad)
  glPopMatrix()

def draw_frame():
  hw, hh = WIDTH/2, HEIGHT/2
  glLineWidth(2.0)
  glColor3f(0.0, 0.8, 1.0)
  glBegin(GL_LINE_LOOP)
  glVertex3f(-hw,-hh,0); glVertex3f( hw,-hh,0)
  glVertex3f( hw, hh,0); glVertex3f(-hw, hh,0)
  glEnd()
  glLineWidth(1.0)

def draw_grid(size=200, step=10):
  glColor3f(0.2, 0.2, 0.2)
  glBegin(GL_LINES)
  for i in range(-size, size+step, step):
    glVertex3f(i,-size,0); glVertex3f(i, size,0)
    glVertex3f(-size,i,0); glVertex3f( size,i,0)
  glEnd()

def draw_axes(length=20):
  glBegin(GL_LINES)
  glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(length,0,0)
  glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,length,0)
  glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,length)
  glEnd()

def draw_sight_line():
  origin = get_pitch_axle_pos()
  hit  = get_ground_hit()
  end  = hit if hit is not None else origin + get_barrel_direction() * 200.0

  # Blue sight line
  glLineWidth(2.0)
  glColor3f(0.1, 0.4, 1.0)
  glBegin(GL_LINES)
  glVertex3f(*origin)
  glVertex3f(*end)
  glEnd()

  # Green line: perpendicular to yaw arm direction, on Z=0 plane
  if hit is not None:
    R    = yaw_rotation_matrix()
    arm_dir  = R @ np.array([1.0, 0.0, 0.0])   # yaw arm points along rotated X
    perp   = np.array([-arm_dir[1], arm_dir[0], 0.0])  # 90deg in XY plane
    perp  /= np.linalg.norm(perp)

    LINE_LEN = 100.0
    p0 = np.array([hit[0], hit[1], 0.01]) - perp * LINE_LEN
    p1 = np.array([hit[0], hit[1], 0.01]) + perp * LINE_LEN

    glColor3f(0.0, 1.0, 0.3)
    glBegin(GL_LINES)
    glVertex3f(*p0)
    glVertex3f(*p1)
    glEnd()

  glLineWidth(1.0)

def draw_rig():
  R     = yaw_rotation_matrix()
  yaw_pos = get_yaw_axle_pos()
  pit_pos = get_pitch_axle_pos()

  # ── Plank (fixed, on top) ─────────────────────────────────────────────────
  draw_cuboid(PLANK_POS[0], PLANK_POS[1], PLANK_POS[2],
        pitch_deg=0, yaw_deg=0,
        width=PLANK_W, depth=PLANK_D, height=PLANK_H,
        color=(0.55, 0.35, 0.15))

  # ── Yaw axle: horizontal rod along X, below +X end of plank ─────────────
  yaw_axis = R @ np.array([1.0, 0.0, 0.0])
  draw_cylinder_along_axis(
    yaw_pos[0], yaw_pos[1], yaw_pos[2],
    axis=yaw_axis,
    length=YAW_AXLE_LEN, radius=YAW_AXLE_RADIUS,
    color=(1.0, 1.0, 1.0)
  )

  # ── Arm: connects yaw axle tip down to pitch servo ────────────────────────
  mid   = (yaw_pos + pit_pos) / 2.0
  delta = pit_pos - yaw_pos
  draw_cylinder_along_axis(
    mid[0], mid[1], mid[2],
    axis=delta,
    length=np.linalg.norm(delta), radius=0.4,
    color=(0.7, 0.7, 0.7)
  )

  # ── Pitch servo body ──────────────────────────────────────────────────────
  draw_cuboid(pit_pos[0], pit_pos[1], pit_pos[2],
        pitch_deg=0, yaw_deg=yServoAngle,
        width=PITCH_BODY_SIZE, depth=PITCH_BODY_SIZE, height=PITCH_BODY_SIZE,
        color=(0.4, 0.4, 0.4))

  # ── Pitch axle: starts along -Z, tilted by pServoAngle around local X ────
  draw_cylinder_along_axis(
    pit_pos[0], pit_pos[1], pit_pos[2],
    axis=get_barrel_direction(),
    length=PITCH_AXLE_LEN, radius=PITCH_AXLE_RADIUS,
    color=(0.55, 0.55, 0.55)
  )

def find_best_yaw_angle():
  global yServoAngle

  saved_yaw = yServoAngle

  best_angle = None
  best_dist  = float('inf')
  results  = []

  for step in range(-1000, 1000, 1):
    deg = step * 90 / 1000 

    yServoAngle = float(deg)

    R     = yaw_rotation_matrix()
    arm_dir = R @ np.array([1.0, 0.0, 0.0])   # yaw arm direction in XY

    # Green line passes through yaw axle pos, perpendicular to arm_dir
    perp  = np.array([-arm_dir[1], arm_dir[0], 0.0])
    perp /= np.linalg.norm(perp)

    # Anchor point on green line = yaw axle projected to Z=0
    yaw_pos = get_yaw_axle_pos()
    anchor  = np.array([yaw_pos[0], yaw_pos[1], 0.0])

    # Distance from POI to infinite line (anchor, perp)
    hp   = np.array([poi_pos[0] - anchor[0], poi_pos[1] - anchor[1], 0.0])
    dist = np.linalg.norm(hp - np.dot(hp, perp) * perp)

    results.append((deg, dist))

    if dist < best_dist:
      best_dist  = dist
      best_angle = deg

  yServoAngle = saved_yaw

  print(f"\n=== Yaw sweep results ===")
  print(f"Best yaw angle : {best_angle}°")
  print(f"Min dist to green line : {best_dist:.4f} units")
  print(f"Top 5 closest angles:")
  for deg, dist in sorted(results, key=lambda x: x[1])[:5]:
    print(f"  {deg:4f}°  ->  dist = {dist:.4f}")

  return best_angle

def find_best_pitch_angle():
  global pServoAngle

  saved_pitch = pServoAngle

  best_angle = None
  best_dist  = float('inf')
  results  = []

  for step in range(-1000, 1000, 1):
    deg = step * 90 / 1000

    pServoAngle = float(deg)

    hit = get_ground_hit()

    if hit is None:
      results.append((deg, float('inf')))
      continue

    # Distance from ground hit point to POI in XY
    dist = np.linalg.norm(
      np.array([hit[0], hit[1]]) - np.array([poi_pos[0], poi_pos[1]])
    )

    results.append((deg, dist))

    if dist < best_dist:
      best_dist  = dist
      best_angle = deg

  pServoAngle = saved_pitch

  print(f"\n=== Pitch sweep results ===")
  print(f"Best pitch angle : {best_angle}°")
  print(f"Min dist hit to POI : {best_dist:.4f} units")
  print(f"Top 5 closest angles:")
  for deg, dist in sorted(results, key=lambda x: x[1])[:5]:
    print(f"  {deg:f}°  ->  dist = {dist:.4f}")

  return best_angle

# ----------------------------------------------------------------------- main

def main():
  global cam_pos, cam_yaw, cam_pitch, poi_pos
  global last_mouse, left_down, right_down
  global yServoAngle, pServoAngle

  pygame.init()
  pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
  pygame.display.set_caption("Servo Visualiser")
  pygame.mouse.set_visible(True)
  pygame.event.set_grab(False)

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(FOV, WINDOW_WIDTH / WINDOW_HEIGHT, NEAR_CLIP, FAR_CLIP)
  glMatrixMode(GL_MODELVIEW)
  glEnable(GL_DEPTH_TEST)
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_COLOR_MATERIAL)
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
  glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 2.0, 0.0])
  glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
  glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3, 1.0])
  glClearColor(0, 0, 0, 1)

  TURN_SPEED = 0.3
  clock    = pygame.time.Clock()
  running  = True

  print("=== Controls ===")
  print("Hold Y / T   : Yaw servo  +/-")
  print("Hold G / H   : Pitch servo +/-")
  print("W/A/S/D    : Move POI (XY plane)")
  print("Right-drag   : Pan  |  Left-drag: Rotate  |  Scroll: Zoom")
  print("Escape     : Quit")

  while running:
    dt    = clock.tick(60) / 1000.0   # seconds
    mx, my  = pygame.mouse.get_pos()

    for event in pygame.event.get():
      if event.type == QUIT:
        running = False
      elif event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
        elif event.key == K_f:        # ← add this
          best = find_best_yaw_angle()
          print(f"→ Applying best yaw angle: {best}°")
          yServoAngle = float(best)     # snap to best angle visually
        elif event.key == K_r:
          best = find_best_pitch_angle()
          print(f"→ Applying best pitch angle: {best}°")
          pServoAngle = float(best)

      elif event.type == MOUSEBUTTONDOWN:
        if   event.button == 3: right_down = True;  last_mouse = (mx, my)
        elif event.button == 1: left_down  = True;  last_mouse = (mx, my)
        elif event.button == 4: cam_pos += get_forward() * SCROLL_SPEED
        elif event.button == 5: cam_pos -= get_forward() * SCROLL_SPEED
      elif event.type == MOUSEBUTTONUP:
        if   event.button == 3: right_down = False; last_mouse = None
        elif event.button == 1: left_down  = False; last_mouse = None
      elif event.type == MOUSEMOTION:
        if last_mouse is not None:
          dx = mx - last_mouse[0]
          dy = my - last_mouse[1]
          if right_down and not left_down:
            pixel_to_world_translation(dx, dy)
          elif left_down and not right_down:
            cam_yaw   += dx * TURN_SPEED
            cam_pitch  = np.clip(cam_pitch - dy * TURN_SPEED, -89.0, 89.0)
          last_mouse = (mx, my)

    # held keys — servo angles updated continuously using dt
    keys = pygame.key.get_pressed()
    if keys[K_y]: yServoAngle += ANGLE_RATE * dt
    if keys[K_t]: yServoAngle -= ANGLE_RATE * dt
    if keys[K_g]: pServoAngle += ANGLE_RATE * dt
    if keys[K_h]: pServoAngle -= ANGLE_RATE * dt

    spd = POI_STEP * dt * 1000.0
    if keys[K_w]: poi_pos[1] += spd
    if keys[K_s]: poi_pos[1] -= spd
    if keys[K_a]: poi_pos[0] -= spd
    if keys[K_d]: poi_pos[0] += spd

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    apply_camera()

    glDisable(GL_LIGHTING)
    draw_grid()
    draw_axes()
    draw_frame()
    draw_sight_line()
    glEnable(GL_LIGHTING)

    draw_rig()

    glDisable(GL_LIGHTING)
    draw_sphere(poi_pos)
    glEnable(GL_LIGHTING)

    pygame.display.flip()

  pygame.quit()

if __name__ == "__main__":
  main()
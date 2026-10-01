import bpy
import json
from pathlib import Path
from mathutils import Vector

q
# Change this to point to your hand_recording.json file.
RECORDING_FILE = Path(
    r"C:\Users\Akira\Downloads\STEM-prototype\recordings\hand_recording.json"
)

FPS = 30.0
SCALE = 8.0
DEPTH_SCALE = 3.0
JOINT_RADIUS = 0.045


# MediaPipe hand landmark connections.
CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


def clear_scene():
    """Delete all objects in the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def landmark_to_blender(landmark):
    """
    Convert MediaPipe landmark coordinates to Blender world space.

    MediaPipe:
      x: 0 to 1 (left to right)
      y: 0 to 1 (top to bottom)
      z: depth (estimated)

    Blender:
      x: left/right
      y: forward/backward
      z: up/down
    """

    x = (landmark["x"] - 0.5) * SCALE
    y = -landmark["z"] * DEPTH_SCALE
    z = (0.5 - landmark["y"]) * SCALE

    return Vector((x, y, z))


def create_joint(index):
    """Create a sphere to represent a hand joint."""

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=12,
        ring_count=8,
        radius=JOINT_RADIUS,
        location=(0, 0, 0),
    )

    joint = bpy.context.object
    joint.name = f"HandJoint_{index:02d}"

    material = bpy.data.materials.get("HandJointMaterial")

    if material is None:
        material = bpy.data.materials.new("HandJointMaterial")
        material.use_nodes = False
        material.diffuse_color = (0.1, 0.7, 1.0, 1.0)

    if material not in joint.data.materials[:]:
        joint.data.materials.append(material)

    return joint


def create_bone_visual(parent_joint, child_joint, index):
    """Create a cylinder connecting two joints."""

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=10,
        radius=JOINT_RADIUS * 0.55,
        depth=1.0,
        location=(0, 0, 0),
    )

    bone = bpy.context.object
    bone.name = f"HandBone_{index:02d}"

    material = bpy.data.materials.get("HandBoneMaterial")

    if material is None:
        material = bpy.data.materials.new("HandBoneMaterial")
        material.use_nodes = False
        material.diffuse_color = (0.05, 0.25, 0.8, 1.0)

    if material not in bone.data.materials[:]:
        bone.data.materials.append(material)

    # Constrain the cylinder to stretch between the two joints.
    stretch_constraint = bone.constraints.new(type="STRETCH_TO")
    stretch_constraint.target = child_joint
    stretch_constraint.head_tail = 0.0
    stretch_constraint.rest_length = 1.0

    copy_location = bone.constraints.new(type="COPY_LOCATION")
    copy_location.target = parent_joint

    return bone


def load_recording(file_path):
    """Load the JSON recording file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Recording file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    data = load_recording(RECORDING_FILE)
    frames = data["frames"]

    print(f"Loading {len(frames)} frames from {RECORDING_FILE}")

    clear_scene()

    # Create 21 joint spheres.
    joints = [
        create_joint(index)
        for index in range(21)
    ]

    # Create bones connecting the joints.
    bones = [
        create_bone_visual(
            joints[parent_index],
            joints[child_index],
            connection_index,
        )
        for connection_index, (parent_index, child_index)
        in enumerate(CONNECTIONS)
    ]

    # Configure the scene.
    scene = bpy.context.scene
    scene.render.fps = int(FPS)
    scene.frame_start = 1
    scene.frame_end = max(1, len(frames))

    # Animate the joints.
    for frame_index, recorded_frame in enumerate(frames, start=1):
        scene.frame_set(frame_index)

        hand = recorded_frame.get("hand")

        # If tracking was lost, hold the previous pose.
        if hand is None:
            continue

        landmarks = hand.get("landmarks", [])

        if len(landmarks) != 21:
            continue

        for landmark_index, landmark in enumerate(landmarks):
            location = landmark_to_blender(landmark)

            joints[landmark_index].location = location
            joints[landmark_index].keyframe_insert(
                data_path="location",
                frame=frame_index,
            )

    # Set all animation curves to linear interpolation.
    for joint in joints:
        if joint.animation_data and joint.animation_data.action:
            for curve in joint.animation_data.action.fcurves:
                for keyframe in curve.keyframe_points:
                    keyframe.interpolation = "LINEAR"

    print(f"Imported {len(frames)} frames successfully.")
    print("Press Spacebar to play the animation.")


main()
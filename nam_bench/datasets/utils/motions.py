import itertools
import numpy as np


def get_random_trajectory(
    seq_length: int=20,
    speed: float=2.5,
    theta: float=0.5,
    img_size: int=30,
    object_size: int=9,
    random_start: bool=False,
) -> np.ndarray:
    """Generate a random trajectory.

    Args:
        seq_length (int, optional): Total sequence(frame) length. Defaults to 20.
        speed (float, optional): Moving speed. Defaults to 2.5.
        theta (float, optional): Moving direction. Defaults to 0.5.
        img_size (int, optional): Image size. Defaults to 30.
        object_size (int, optional): Object size. Defaults to 9.
        random_start (bool, optional): Random start position. Defaults to False.
    """
    # Initialize the trajectory
    trajectory = np.zeros((seq_length, 2), dtype=np.int32)
    # Initialize the position of the object
    if random_start:
        position = np.random.randint(0, img_size - object_size, size=2)
    else:
        position = np.array([(img_size-object_size)//2, (img_size - object_size)//2])
    # Initialize the speed of the object
    vx, vy = np.cos(theta*2*np.pi), np.sin(theta * 2*np.pi)
    vx, vy = int(speed * vx), int(speed * vy)

    # Loop over the frames
    for t in range(seq_length):
        # Update the position of the object
        trajectory[t] = position
        # Update the position of the object
        position = position + np.array([vx, vy])
        # Check if the object is outside of the image
        if position[0] <= 0:
            vx = -vx
            position[0] = 0
        if position[0] >= img_size - object_size:
            vx = -vx
            position[0] = img_size - object_size - 1
        if position[1] <= 0:
            vy = -vy
            position[1] = 0
        if position[1] >= img_size - object_size:
            vy = -vy
            position[1] = img_size - object_size - 1

    return trajectory


def get_sequences(
    seq_length: int=20,
    img_size: int=30, 
    speed: float=2.5, 
    theta: float=0.0, 
    objects: np.ndarray|None=None
) -> np.ndarray:
    """Generate a sequence of images.

    Args:
        seq_length (int, optional): Sequence length. Defaults to 20.
        img_size (int, optional): Image size. Defaults to 30.
        speed (float, optional): Object speed. Defaults to 2.5.
        theta (float, optional): Direction. Defaults to 0.0.
        objects (np.ndarray, optional): Object to move. Defaults to None.

    Returns:
        np.ndarray: Batch of sequence.
    """
    batch_of_sequence = np.zeros((len(objects), seq_length, img_size, img_size), dtype=np.float32)
    for object_idx, object in enumerate(objects):
        sequence = np.zeros((seq_length, img_size, img_size), dtype=np.float32)
        
        trajectory = get_random_trajectory(seq_length, speed, theta, img_size, object.shape[0])
        h, w = object.shape
        
        for t in range(seq_length):
            x, y = trajectory[t]
            x, y = int(x), int(y)
            sequence[t, y:y+h, x:x+w] = object
    
        batch_of_sequence[object_idx] = sequence
    
    return batch_of_sequence
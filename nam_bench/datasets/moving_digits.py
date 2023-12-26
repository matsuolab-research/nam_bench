import math
import itertools
import random

import numpy as np
import torch
from PIL import Image
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .utils import motions


# 並進最大速度に当たるspeedを3.0から2.0に変更（5x5のパッチサイズでタスクを進めるため）
def generate_moving_toyproblem_imgs(
    n_frames_total: int = 20,
    img_size: int = 30,
    speed: float = 2.5,
    theta: float = 0.0,
    object_array: np.ndarray | None = None,
    obj_scales: list[float] = [1.0, 1.25, 1.5, 1.25], 
    rotation_speed: int | float =30, 
    random_start: bool = False
) -> list[np.ndarray]:
    """Generate moving toyproblem images

    Args:
        n_frames_total (int, optional): フレーム数. Defaults to 20.
        img_size (int, optional): 画像サイズ. Defaults to 30.
        speed (float, optional): オブジェクトの並進速度. Defaults to 2.5.
        theta (float, optional): オブジェクトの回転角. Defaults to 0.0.
        object_array (np.ndarray | None, optional): オブジェクトを格納した配列. Defaults to None.
        obj_scales (list[float], optional): オブジェクトのスケール. Defaults to [1.0, 1.25, 1.5, 1.25].
        rotation_speed (int | float, optional): 角速度. Defaults to 30.
        random_start (bool, optional): 初期位置をランダムにするか,中心から始めるか. Defaults to False.

    Returns:
        list[np.ndarray]: _description_
    """

    max_scale = max(obj_scales)
    data_list = [None] * object_array.shape[0]
    padding = int(img_size * (max_scale-1.0))
    half_padding = int(padding//2)
    # print("Debug", padding, half_padding)
    for obs_index, obs in enumerate(object_array):
        data = np.zeros((n_frames_total, img_size+padding, img_size+padding), dtype=np.float32)  # [20, 30, 30]
        
        # [1] 軌跡（指定されたパラメータで移動するときの物体の中心座標）を取得
        trajectory = motions.get_random_trajectory(
            seq_length=n_frames_total,
            speed=speed,
            theta=theta,
            img_size=img_size,
            object_size=object_array.shape[1],
            random_start=random_start
        )
        start_y, start_x = trajectory[:, 0], trajectory[:, 1]

        # [2] objectを取得
        obs_size = len(obs)

        max_object_size = math.ceil(max_scale*obs_size)
        # print("[DEBUG]", "max_object_size", max_object_size)  # [DEBUG]
        obs = Image.fromarray(obs)

        # [3] objectを描画
        for i in range(n_frames_total):
            # [3-1] Rotate
            # 90度以外でこれを使うと解像度低すぎて画像がかなり変な感じになるので非推奨．90度回転も激しすぎる気がするが．
            angle = (rotation_speed * i) % 360
            obs = obs.rotate(angle, resample=Image.NEAREST)

            # [3-2] Scaling
            current_size = int(obs_size * obj_scales[i])
            # current_size = int(obs_size * scale_fn(i*2*np.pi / scaling_speed))
            tmp_top = int(start_y[i]) + padding // 2
            tmp_left = int(start_x[i]) + padding // 2
            mid_y = tmp_top + math.ceil(max_object_size / 2)
            mid_x = tmp_left + math.ceil(max_object_size / 2)
            
            top = mid_y - current_size // 2
            left = mid_x - current_size // 2
            bottom = top + current_size
            right = left + current_size

            obs = obs.resize(
                (current_size, current_size), resample=Image.NEAREST)
            data[i, top:bottom, left:right] = obs

        if padding == 0:
            data_list[obs_index] = data # S, H, W
        else:
            data_list[obs_index] = data[:, half_padding:-half_padding, half_padding:-half_padding] # S, H, W
    
    return data_list # len(object_array), S, H, W


def load(
    num_train_data: int = 10, 
    num_test_data: int = 10, 
    img_size: int = 20, 
    thetas: list[float] = [0.0, 0.25, 0.5, 0.75], 
    speeds: list[float] = [2.0, 4.0], 
    seq_length: int = 4, 
    binarize: bool = True, 
    obj_scales: list[float] = [1.0, 1.25, 1.5, 1.25], 
    obj_rotation_speeds: list[int] = [30], 
    random_start: bool = False
) -> dict[str, dict[str, np.ndarray | list[dict]]]:
    """ Moving digits dataset

    Args:
        num_train_data (int, optional): 訓練に使う物体の種類．
        num_test_data (int, optional): テストに使う物体の種類．
        img_size (int, optional): キャンバスのサイズ．
        thetas (List[float], optional): 物体が進む方向.2*theta*piの方向に進む.
        speeds (List[float], optinoal): 物体が進む速度．
        seq_length (int, optional): 系列長．
        binarize (bool, optional): 2値化するか
        obj_scales (List[float]): スケール変化の周期
        obj_rotation_speeds (List[float]): 各ステップでどれくらい回転させるか．
        random_start (bool): 初期値を中心に固定するかどうか（しない場合はランダム）
    """


    def preprocess(image):
        image = image / 15
        if binarize:
            image = image = (image > 0.5).astype(float)
        return image

    # [1] Load Digits training and test dataset
    digits = datasets.load_digits()

    # [1-1] Training Dataset
    train_images, test_images, train_labels, test_labels = train_test_split(
        digits.images,
        digits.target,
        train_size=num_train_data,
        test_size=num_test_data,
        random_state=0,
        stratify=digits.target,
    )
    train_images_and_labels = list(zip(train_images, train_labels))
    X_train = []
    for index, (image, label) in enumerate(train_images_and_labels):
        X_train.append(preprocess(image))
    X_train = np.array(X_train)

    # [1-1] Test Dataset
    test_images_and_labels = list(zip(test_images, test_labels))
    X_test = []
    for index, (image, label) in enumerate(test_images_and_labels):
        X_test.append(preprocess(image))
    X_test = np.array(X_test)

    # [2] Generate moving digits datasets
    # [2-1] Generate training datasets
    X_train_seq = []
    metainfo_train = []
    for theta, speed, rotation_speed in itertools.product(thetas, speeds, obj_rotation_speeds):
        videos = generate_moving_toyproblem_imgs(
            object_array=X_train, speed=speed, theta=theta, img_size=img_size, n_frames_total=seq_length,
            obj_scales=obj_scales, rotation_speed=rotation_speed, random_start=random_start)
        X_train_seq += videos # list[np.ndarray]: len(X_train), (S, H, W)
        metainfo_train += [
            {"target": str(int(label)), "speed": speed, "theta": theta, "obj_scales": obj_scales, "rotation_speed": rotation_speed} 
            for label in train_labels
        ]
    X_train_seq = np.stack(X_train_seq)
    y_train = X_train_seq.copy()[:, -1]
    X_train_seq[:, -1] = 0.0

    # [2-2] Generate test datasets
    X_test_seq = []
    metainfo_test = []
    for theta, speed, rotation_speed in itertools.product(thetas, speeds, obj_rotation_speeds):
        videos = generate_moving_toyproblem_imgs(
            object_array=X_test, speed=speed, theta=theta, img_size=img_size, n_frames_total=seq_length,
            obj_scales=obj_scales, rotation_speed=rotation_speed, random_start = random_start)
        X_test_seq += videos
        metainfo_test += [
            {"target": str(int(label)), "speed": speed, "theta": theta, "obj_scales": obj_scales, "rotation_speed": rotation_speed} 
            for label in train_labels
        ]
    X_test_seq = np.stack(X_test_seq)
    y_test = X_test_seq.copy()[:, -1]
    X_test_seq[:, -1] = 0.0
    
    return {
        "train": {
            "X": X_train_seq,
            "y": y_train,
            "metainfo": metainfo_train,
        },
        "test": {
            "X": X_test_seq,
            "y": y_test,
            "metainfo": metainfo_test,
        }
    }


def load_from_config(dataset_config) -> dict[str, dict[str, np.ndarray | list[dict]]]:
    """Load Moving Digits dataset from config

    Args:
        config : Config object

    Returns:
        dict[str, dict[str, np.ndarray | list[dict]]]: Dataset dict
    """
    return load(
        num_train_data=dataset_config.num_train_data,
        num_test_data=dataset_config.num_test_data,
        img_size=dataset_config.img_size,
        thetas=dataset_config.thetas,
        speeds=dataset_config.speeds,
        seq_length=dataset_config.seq_length,
        binarize=dataset_config.binarize,
        obj_scales=dataset_config.obj_scales,
        obj_rotation_speeds=dataset_config.obj_rotation_speeds,
        random_start=dataset_config.random_start,
    )
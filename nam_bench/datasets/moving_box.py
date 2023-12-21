from functools import partial

import numpy as np


class ObjectOrbit:
    """Calculate object orbit
    """
    def __init__(self, base_len: int|float, obj_len: int|float, omega: float=None, T: float=None, phase: float=0.0):
        """Initialize

        Args:
            base_len (int | float): オブジェクトが移動できる範囲の長さ
            obj_len (int | float): オブジェクトの長さ
            omega (float, optional): 各速度. Defaults to None.
            T (float, optional): 周期. Defaults to None.
            phase (float, optional): 位相. Defaults to 0.0.

        Raises:
            ValueError: 周期と速度のどちらか一方を指定する必要があります
            ValueError: オブジェクトの長さは移動できる範囲の長さよりも短くする必要があります
            ValueError: 周期は2以上でなければなりません. また, 角速度はpi以下でなければなりません
        """
        if (omega is None and T is None) or (omega is not None and T is not None):
            raise ValueError("One of omega and T must be specified")
        if obj_len > base_len:
            raise ValueError("Object length must be smaller than base length")
        self.base_len = base_len
        self.obj_len = obj_len
        self.phase = phase
        if omega is None:
            self.omega = 2 * np.pi / T
            self.T = T
        else:
            self.T = 2 * np.pi / omega
            self.omega = omega

        if self.T <= 2:
            raise ValueError(f"T must be larger than or equal to 2 {self.T} and omega must be smaller than or equal to pi {self.omega}.\nThis restriction originates from aliasing")
        self.amplitude = (base_len - obj_len) / 2

    def generate_relative_orbit(self, seq_len: int) -> np.ndarray:
        """Generate relative orbit

        Args:
            seq_len (int): 系列長

        Returns:
            np.ndarray: 相対軌道
        """
        relative_orbit = np.zeros(seq_len, dtype=np.float32)
        relative_orbit = self.amplitude*np.sin(self.omega * np.arange(seq_len) + self.phase)

        return relative_orbit

    def generate_orbit(self, base_orbit: np.ndarray) -> np.ndarray:
        return base_orbit + self.generate_relative_orbit(len(base_orbit))

    def generate_orbit_left(self, base_orbit: np.ndarray):
        return base_orbit + self.generate_relative_orbit(len(base_orbit)) - self.obj_len/2


####################################################################################################
# 軌道生成のための最低限の関数
def get_trajectory(
    seq_len: int,
    frame_size: tuple[int],
    num_objs: int,
    obj_heights: np.ndarray,
    obj_widths: np.ndarray,
    periodic_times: np.ndarray
) -> tuple[np.ndarray]:
    """Get trajectory

    Args:
        seq_len (int): 系列長
        frame_size (tuple[int]): 画像サイズ
        num_objs (int): オブジェクト数
        obj_heights (np.ndarray): 各オブジェクトの高さ
        obj_widths (np.ndarray): 各オブジェクトの幅
        periodic_times (np.ndarray): 各オブジェクトの周期
    Raises:
        ValueError: obj_widthsは降順にソートされている必要があります

    Returns:
        tuple[np.ndarray]: 軌跡, 各オブジェクトの幅, 各オブジェクトの高さ, 各オブジェクトの周期
    """
    if not np.all(obj_widths == np.sort(obj_widths)[::-1]):
        raise ValueError("obj_widths must be sorted in descending order.")
    height_sum = obj_heights.sum()
    if height_sum > frame_size[0]: # オブジェクトの高さの合計が画像の高さを超える場合は正規化
        obj_heights /= height_sum/frame_size[0]
        obj_heights = obj_heights.astype(int)

    obj_orbits = np.zeros((num_objs, seq_len), dtype=np.float32)
    shifted_obj_widths = np.roll(obj_widths, 1)
    shifted_obj_widths[0] = frame_size[1] # base_lenを渡すために調整
    for i in range(num_objs):
        obj_orbit = ObjectOrbit(shifted_obj_widths[i], obj_widths[i], T=periodic_times[i])
        obj_orbits[i] = obj_orbit.generate_relative_orbit(seq_len) # 相対座標のみ取得
    base_orbits = np.full(seq_len, frame_size[1]/2) # 1つ目のobjectの基準となる点の設定
    obj_orbits[0] += base_orbits # 1つ目のobjectを絶対座標に変換
    obj_orbits = np.cumsum(obj_orbits, axis=0) # 以降のobjectも絶対座標に変換

    return obj_orbits, obj_widths, obj_heights, periodic_times


# NOTE: ここで部分適用する関数を中で作る
def get_random_freq_trajectories(
    sample_size: int,
    seq_len: int,
    frame_size: tuple[int],
    num_objs: int,
    obj_heights: np.ndarray,
    obj_widths: np.ndarray,
    min_periodic_time: float, max_periodic_time: float,
) -> tuple[np.ndarray]:
    """Get random frequency trajectories while keeping the object size

    Args:
        sample_size (int): サンプル数(=バッチサイズ)
        seq_len (int): 系列長
        frame_size (tuple[int]): 画像サイズ
        num_objs (int): オブジェクト数
        obj_heights (np.ndarray): オブジェクトの高さ
        obj_widths (np.ndarray): オブジェクトの幅
        min_periodic_time (float): ランダムに選ぶ中での最小の周期
        max_periodic_time (float): ランダムに選ぶ中での最大の周期

    Returns:
        tuple[np.ndarray]: 軌跡, 各オブジェクトの幅, 各オブジェクトの高さ, 各オブジェクトの周期
    """
    periodic_times = np.random.uniform(min_periodic_time, max_periodic_time, (sample_size, num_objs))
    # NOTE: 引数の一部を固定した関数を作成
    get_random_freq_trajectory = partial(
        get_trajectory, 
        seq_len=seq_len, 
        frame_size=frame_size,
        num_objs=num_objs,
        obj_heights=obj_heights,
        obj_widths=obj_widths
    )

    obj_orbits = np.zeros((sample_size, num_objs, seq_len), dtype=np.float32)
    obj_widths = np.zeros((sample_size, num_objs), dtype=int)
    obj_heights = np.zeros((sample_size, num_objs), dtype=int)
    for i in range(sample_size):
        obj_orbits[i], obj_widths[i], obj_heights[i], _ = get_random_freq_trajectory(
            periodic_times=periodic_times[i],
        )

    return obj_orbits, obj_widths, obj_heights, periodic_times


def get_random_trajectories(
    sample_size: int,
    seq_len: int,
    frame_size: tuple[int],
    num_objs: int,
    min_height: int, max_height: int,
    min_width: int, max_width: int,
    min_periodic_time: float, max_periodic_time: float,
    ratio: float | np.ndarray | None = None,
) -> tuple[np.ndarray]:
    """Get random trajectories while keeping the number of objects.

    Args:
        sample_size (int): サンプル数(=バッチサイズ)
        seq_len (int): 系列長
        frame_size (tuple[int]): 画像サイズ
        num_objs (int): オブジェクト数
        min_height (int): オブジェクトの高さの最小値
        max_height (int): オブジェクトの高さの最大値
        min_width (int): オブジェクトの幅の最小値
        max_width (int): オブジェクトの幅の最大値
        min_periodic_time (float): ランダムに選ぶ中での最小の周期
        max_periodic_time (float): ランダムに選ぶ中での最大の周期
        ratio (float | np.ndarray | None, optional): オブジェクトの幅の比率. Defaults to None.

    Returns:
        tuple[np.ndarray]: 軌跡, 各オブジェクトの幅, 各オブジェクトの高さ, 各オブジェクトの周期
    """
    if max_width > frame_size[1]:
        raise ValueError("max_width must be less than frame_size[1]")
    if max_height > frame_size[0]:
        raise ValueError("max_height must be less than frame_size[0]")
    
    # 各種パラメータの設定
    ####################################################################################################
    # NOTE: for文の中で設定しても問題ない.(そっちの方が可読性が高いかも？)
    obj_heights = np.random.randint(min_height, max_height, size=(sample_size, num_objs))
    height_sum = obj_heights.sum(axis=1)
    over_height_limit_idx = height_sum > frame_size[0]
    obj_heights = obj_heights.astype(np.float32)
    if np.any(over_height_limit_idx):
        obj_heights[over_height_limit_idx] /= height_sum[over_height_limit_idx]/frame_size[0]
    obj_heights = obj_heights.astype(int)

    if ratio is None:
        obj_widths = np.random.randint(min_width, max_width, size=(sample_size, num_objs))
        obj_widths = np.sort(obj_widths, axis=1)[:, ::-1]
    else:
        exp = np.arange(1, num_objs+1)
        obj_widths = frame_size[1] * np.power(ratio, exp)
        obj_widths = obj_widths.astype(int)
        if obj_widths[-1] == 0:
            raise ValueError("width_ratio or frame_size is too small so that the smallest object width become 0.")
        obj_widths = np.tile(obj_widths, (sample_size, 1))
        
    periodic_times = np.random.uniform(min_periodic_time, max_periodic_time, size=(sample_size, num_objs))
    ####################################################################################################
    
    get_random_trajectory = partial(
        get_trajectory,
        seq_len=seq_len,
        frame_size=frame_size,
        num_objs=num_objs,
    )
    
    obj_orbits = np.zeros((sample_size, num_objs, seq_len), dtype=np.float32)
    for i in range(sample_size):
        obj_orbits[i], _, _, _ = get_random_trajectory(
            obj_heights=obj_heights[i],
            obj_widths=obj_widths[i],
            periodic_times=periodic_times[i],
        )
    
    return obj_orbits, obj_widths, obj_heights, periodic_times


####################################################################################################
# 描画系
####################################################################################################
def render_trajectory(
    frame_size: tuple[int],
    obj_orbits: np.ndarray,
    obj_widths: np.ndarray,
    obj_heights: np.ndarray,
    periodic_times: np.ndarray,
    colors: int | np.ndarray=255
) -> np.ndarray:
    """Render a sample of trajectory

    Args:
        frame_size (tuple[int]): 画像サイズ
        obj_orbits (np.ndarray): オブジェクトの軌跡
        obj_widths (np.ndarray): オブジェクトの幅
        obj_heights (np.ndarray): オブジェクトの高さ
        periodic_times (np.ndarray): オブジェクトの周期
        colors (int | np.ndarray, optional): オブジェクトの色. Defaults to 255.

    Returns:
        np.ndarray: 描画された画像(S, H, W)
    """
    num_objs, seq_len = obj_orbits.shape
    if isinstance(colors, int):
        colors = np.full(num_objs, colors)

    # objectの左端の座標
    obj_orbits_left = obj_orbits - obj_widths.reshape(-1, 1)/2 # 中心座標 -> 左側の座標
    obj_orbits_left = obj_orbits_left.astype(int)
    # objectの高さの座標
    obj_bottoms = np.zeros(num_objs, dtype=int)
    obj_bottoms[1:] = obj_heights[:-1].cumsum()

    canvas = np.zeros((seq_len, *frame_size))
    for obj_idx in range(num_objs):
        lefts = obj_orbits_left[obj_idx]
        rights = obj_orbits_left[obj_idx]+obj_widths[obj_idx]
        bottom = obj_bottoms[obj_idx]
        top = obj_bottoms[obj_idx]+obj_heights[obj_idx]
        for seq_idx, (left, right) in enumerate(zip(lefts, rights)):
            canvas[seq_idx, bottom:top, left:right] = colors[obj_idx]

    return canvas


def render_trajectories(
    frame_size: tuple[int],
    obj_orbits: np.ndarray,
    obj_widths: np.ndarray,
    obj_heights: np.ndarray,
    periodic_times: np.ndarray,
    colors: np.ndarray | int = 255
) -> np.ndarray:
    """Render trajectories

    Args:
        frame_size (tuple[int]): 画像サイズ
        obj_orbits (np.ndarray): オブジェクトの軌道
        obj_widths (np.ndarray): オブジェクトの幅
        obj_heights (np.ndarray): オブジェクトの高さ
        periodic_times (np.ndarray): オブジェクトの周期
        colors (np.ndarray | int, optional): オブジェクトの色. Defaults to 255.

    Returns:
        np.ndarray: 描画された画像(B, S, H, W)
    """
    sample_num, num_objs, seq_len = obj_orbits.shape
    if isinstance(colors, int):
        colors = np.full((sample_num, num_objs), colors, dtype=np.uint8)

    canvas = np.zeros((sample_num, seq_len, *frame_size), dtype=np.uint8)
    for sample_idx in range(sample_num):
        canvas[sample_idx] = render_trajectory(
            frame_size,
            obj_orbits[sample_idx],
            obj_widths[sample_idx],
            obj_heights[sample_idx],
            periodic_times[sample_idx],
            colors[sample_idx],
        )

    return canvas


def get_random_orbit_imgs(
        sample_size: int,
        seq_len: int,
        frame_size: tuple[int],
        num_objs: int,
        min_height: int, max_height: int,
        min_width: int, max_width: int,
        min_periodic_time: float, max_periodic_time: float,
        color: str|int = 255,
        ratio: float | np.ndarray | None = None,
) -> np.ndarray:
    """Get random orbit images

    Args:
        sample_size (int): サンプル数
        seq_len (int): 系列長
        frame_size (tuple[int]): 画像サイズ
        num_objs (int): オブジェクト数
        min_height (int): オブジェクトの高さの最小値
        max_height (int): オブジェクトの高さの最大値
        min_width (int): オブジェクトの幅の最小値
        max_width (int): オブジェクトの幅の最大値
        min_periodic_time (float): オブジェクトの周期の最小値
        max_periodic_time (float): オブジェクトの周期の最大値
        color (str | int, optional): オブジェクトの色. Defaults to 255.
        ratio (float | np.ndarray | None, optional): 隣接オブジェクトの幅の比. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    if color == "random":
        colors = np.random.randint(1, 255, size=(sample_size, num_objs), dtype=np.uint8)
    elif color == "alternate":
        colors = np.full((sample_size, num_objs), 255, dtype=np.uint8)
        colors[:, ::2] = 128
    else:
        colors = np.full((sample_size, num_objs), color, dtype=np.uint8)

    obj_orbits, obj_widths, obj_heights, periodic_times = get_random_trajectories(
        sample_size,
        seq_len,
        frame_size,
        num_objs,
        min_height, max_height,
        min_width, max_width,
        min_periodic_time, max_periodic_time,
        ratio,
    )
    canvas = render_trajectories(frame_size, obj_orbits, obj_widths, obj_heights, periodic_times, colors)
    return canvas


####################################################################################################
# 実際に呼び出す関数
def load(
    frame_size: tuple[int] = (100, 100),
    num_train_data: int = 10,
    num_test_data: int = 10,
    seq_len: int = 20,
    num_objs: int = 4,
    min_height: int = 5, max_height: int = 15,
    min_width: int = 15, max_width: int = 75,
    fix_obj_widths: tuple[int] | np.ndarray | None = None,
    fix_obj_heights: tuple[int] | np.ndarray | None = None,
    min_periodic_time: float = 4, max_periodic_time: float = 80,
    color: str | int = "alternate",
    ratio: float | np.ndarray | None = None,
    random_objs: bool = False,
    image: bool = True,
) -> tuple[np.ndarray]:
    """Load moving box dataset
    TODO: fix_parameters (dict[str, Any])で,一部のパラメータを固定できるようにする.

    Args:
        frame_size (tuple[int], optional): 画像サイズ. Defaults to (100, 100).
        num_train_data (int, optional): 訓練に使う物体の種類. Defaults to 10.
        num_test_data (int, optional): テストに使う物体の種類. Defaults to 10.
        seq_len (int, optional): 系列長. Defaults to 20.
        num_objs (int, optional): 1つの系列に出てくるオブジェクトの数. Defaults to 4.
        min_height (int, optional): オブジェクトの最小の高さ(random_obj指定時に利用). Defaults to 5.
        max_height (int, optional): オブジェクトの最大の高さ(random_obj指定時に利用). Defaults to 15.
        min_width (int, optional): オブジェクトの最小の幅(random_obj指定時に利用). Defaults to 15.
        max_width (int, optional): オブジェクトの最大の幅(random_obj指定時に利用). Defaults to 75.
        fix_obj_widths (tuple[int] | np.ndarray | None, optional): オブジェクトの幅(random_obj未指定時に必須). Defaults to None.
        fix_obj_heights (tuple[int] | np.ndarray | None, optional): オブジェクトの高さ(random_obj未指定時に必須). Defaults to None.
        min_periodic_time (float, optional): 最小の周期(random_obj指定時に利用). Defaults to 4.
        max_periodic_time (float, optional): 最大の周期(random_obj指定時に利用). Defaults to 80.
        color (str | int, optional): オブジェクトの色. Defaults to "alternate".
        ratio (float | np.ndarray | None, optional): オブジェクトのサイズの比を固定するか？ex) 100, 50, 25, 12.5.... Defaults to None.
        random_objs (bool, optional): オブジェクトの属性をランダムに生成するか？(幅、高さ、周期). Defaults to False.
        image (bool, optional): 動画像として返すか,軌道のみを返すか. Defaults to True.

    Returns:
        tuple[np.ndarray]: _description_
    """
    if not random_objs and (fix_obj_heights is None or fix_obj_widths is None):
        raise ValueError("fix_obj_heights and fix_obj_widths must be specified when random_objs is False")
    if not random_objs and (len(fix_obj_heights) != num_objs or len(fix_obj_widths) != num_objs):
        raise ValueError("fix_obj_heights and fix_obj_widths must be the same length as num_objs")
    
    sample_size = num_train_data + num_test_data

    if random_objs:
        obj_orbits, obj_widths, obj_heights, periodic_times = get_random_trajectories(
            sample_size=sample_size,
            seq_len=seq_len,
            frame_size=frame_size,
            num_objs=num_objs,
            min_height=min_height, max_height=max_height,
            min_width=min_width, max_width=max_width,
            min_periodic_time=min_periodic_time, max_periodic_time=max_periodic_time,
            ratio=ratio,
        )
    else:
        obj_orbits, obj_widths, obj_heights, periodic_times = get_random_freq_trajectories(
            sample_size=sample_size,
            seq_len=seq_len,
            frame_size=frame_size,
            num_objs=num_objs,
            obj_heights=fix_obj_heights,
            obj_widths=fix_obj_widths,
            min_periodic_time=min_periodic_time, max_periodic_time=max_periodic_time,
        )
    
    metainfo = [None]*sample_size
    for i, (obj_width, obj_height, periodic_time) in enumerate(zip(obj_widths, obj_heights, periodic_times)):
        metainfo[i] = {
            "width": obj_width,
            "height": obj_height,
            "periodic_time": periodic_time,
        }
    if not image:
        train_test_dataset = {
            "train": {
                "X": obj_orbits[:num_train_data, ..., :-1],
                "y": obj_orbits[:num_train_data, ..., -1],
                "metainfo": metainfo[:num_train_data],
            },
            "test": {
                "X": obj_orbits[num_train_data:, ..., :-1],
                "y": obj_orbits[num_train_data:, ..., -1],
                "metainfo": metainfo[num_train_data:],
            },
        }
        return train_test_dataset
    
    # 以下レンダリング
    if color == "random":
        colors = np.random.randint(1, 255, size=(sample_size, num_objs), dtype=np.uint8)
    elif color == "alternate":
        colors = np.full((sample_size, num_objs), 255, dtype=np.uint8)
        colors[:, ::2] = 128
    else:
        colors = np.full((sample_size, num_objs), color, dtype=np.uint8)
    
    canvas = render_trajectories(
        frame_size,
        obj_orbits,
        obj_widths,
        obj_heights,
        periodic_times,
        colors,
    )
    
    train_test_dataset = {
        "train": {
            "X": canvas[:num_train_data, :-1],
            "y": canvas[:num_train_data, -1],
            "metainfo": metainfo[:num_train_data],
        },
        "test": {
            "X": canvas[num_train_data:, :-1],
            "y": canvas[num_train_data:, -1],
            "metainfo": metainfo[num_train_data:],
        },
    }
    return train_test_dataset
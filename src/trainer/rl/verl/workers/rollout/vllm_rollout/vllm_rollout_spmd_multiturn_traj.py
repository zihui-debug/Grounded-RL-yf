import os
import re
from contextlib import contextmanager
from typing import Any, List, Union, Tuple, Optional
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from PIL import Image, ImageDraw
import json
import copy
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from copy import deepcopy
# import logging
import logging
logger = logging.getLogger(__name__)

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 576 * 28 * 28
IMAGE_FACTOR = 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

@dataclass
class VideoProcessor:
    max_frames: int = 32
    max_frame_pixels: int = 224*224
    min_frame_pixels: int = 64*64
    image_root: Optional[str] = None
    processor: Optional[ProcessorMixin] = None

    def process_video(self, video_info):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 1
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_info)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_info)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

        torchvision_video = None
        try:
            torchvision_video = self.video_torchvision(video_info)
            return torchvision_video
        except Exception as e:
            print(f"torchvision attempt failed: {e}")

    def calculate_video_frame_range(
        self,
        ele: dict,
        total_frames: int,
        video_fps: float,
    ) -> tuple[int, int, int]:
        """
        Calculate the start and end frame indices based on the given time range.

        Args:
            ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
            total_frames (int): Total number of frames in the video.
            video_fps (float): Frames per second of the video.

        Returns:
            tuple: A tuple containing (start_frame, end_frame, frame_count).

        Raises:
            ValueError: If input parameters are invalid or the time range is inconsistent.
        """
        # Validate essential parameters
        if video_fps <= 0:
            raise ValueError("video_fps must be a positive number")
        if total_frames <= 0:
            raise ValueError("total_frames must be a positive integer")

        # Get start and end time in seconds
        video_start = ele.get("video_start", None)
        video_end = ele.get("video_end", None)
        if video_start is None and video_end is None:
            return 0, total_frames - 1, total_frames

        max_duration = total_frames / video_fps
        # Process start frame
        if video_start is not None:
            video_start_clamped = max(0.0, min(video_start, max_duration))
            start_frame = math.ceil(video_start_clamped * video_fps)
        else:
            start_frame = 0
        # Process end frame
        if video_end is not None:
            video_end_clamped = max(0.0, min(video_end, max_duration))
            end_frame = math.floor(video_end_clamped * video_fps)
            end_frame = min(end_frame, total_frames - 1)
        else:
            end_frame = total_frames - 1

        # Validate frame order
        if start_frame >= end_frame:
            raise ValueError(
                f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
                f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
                f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
            )

        # logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
        return start_frame, end_frame, end_frame - start_frame + 1
    

    def smart_nframes(
        self,
        ele: dict,
        total_frames: int,
        video_fps: int | float,
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            fps = ele.get("fps", FPS)
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
            nframes = total_frames / video_fps * fps
            if nframes > total_frames:
                logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
            nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
            nframes = floor_by_factor(nframes, FRAME_FACTOR)
        nframes = min(nframes, total_frames)
        if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
        return nframes

    def video_decord(self, video_info):
        from decord import VideoReader
        video_file = video_info['video']
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()

        start_frame, end_frame, total_frames = self.calculate_video_frame_range(
            video_info,
            total_frames,
            avg_fps,
        )
        video_length = total_frames / avg_fps

        nframes = self.smart_nframes(video_info, total_frames=total_frames, video_fps=avg_fps)
        frame_idx = np.linspace(start_frame, end_frame, nframes, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()

        return self.process_video_frames(video_info, video, frame_idx, video_length)

    def video_torchcodec(self, video_info):
        from torchcodec.decoders import VideoDecoder
        device = "cpu"  # or e.g. "cuda"
        video_file = video_info['video']
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps

        start_frame, end_frame, total_frames = self.calculate_video_frame_range(
            video_info,
            total_frames,
            avg_fps,
        )
        video_length = total_frames / avg_fps

        nframes = self.smart_nframes(video_info, total_frames=total_frames, video_fps=avg_fps)
        frame_idx = np.linspace(start_frame, end_frame, nframes, dtype=int)
        frame_idx = np.unique(frame_idx)

        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().permute(0, 2, 3, 1).numpy()
        return self.process_video_frames(video_info, video, frame_idx, video_length)

    def video_torchvision(self, video_info):
        import torchvision
        video_file = video_info["video"]
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")

        # 读取视频 (T, H, W, C) 默认格式
        video, audio, info = torchvision.io.read_video(
            video_file,
            start_pts=video_info.get("video_start", 0.0),
            end_pts=video_info.get("video_end", None),
            pts_unit="sec",
            output_format="THWC",   # 和 decord 的输出更一致
        )

        total_frames, avg_fps = video.size(0), info["video_fps"]
        video_length = total_frames / avg_fps

        # 采样帧数
        nframes = self.smart_nframes(video_info, total_frames=total_frames, video_fps=avg_fps)
        frame_idx = np.linspace(0, total_frames-1, nframes, dtype=int)
        frame_idx = np.unique(frame_idx)

        # 按 index 取帧
        video = video[frame_idx]  # (T, H, W, C) tensor
        video = video.numpy()     # 转成 numpy，和 decord 结果保持一致

        return self.process_video_frames(video_info, video, frame_idx, video_length)

    def process_video_frames(self, video_info, video, frame_idx, video_length):
        import torch.nn.functional as F
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.processor.image_processor)
        processor.max_pixels = video_info.get("max_pixels", self.max_frame_pixels)
        processor.min_pixels = video_info.get("min_pixels", self.min_frame_pixels)
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]

        T, H_grid, W_grid = grid_thw.tolist()
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        video_tensor = F.interpolate(video_tensor, size=(H_grid*processor.patch_size, W_grid*processor.patch_size), mode="bilinear", align_corners=False)
        video_tensor = video_tensor.permute(0, 2, 3, 1) # (T, H, W, C)

        second_per_grid_ts = [
            self.processor.image_processor.temporal_patch_size / fps
        ] * len(video_processed["video_grid_thw"])
        second_per_grid_ts = torch.tensor(second_per_grid_ts)
        return video_tensor.numpy(), grid_thw, second_per_grid_ts

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def get_eos_mask_last(response_ids: torch.Tensor, eos_token_id: Union[int, List[int]] = 2, dtype: torch.dtype = torch.long):
    """Get the mask for the response ids, the mask will be 0 after the last eos token.

    eos_token_id can be int or list: 1 or [1, 2].
    ```
    e.g. eos_token = 1
    response_ids: [0, 0, 2, 4, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ```
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    # Create a mask that marks all EOS token positions
    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_id:
        eos_mask |= response_ids.eq(token_id)
    
    # For each sequence in the batch, find the position of the last EOS token
    batch_size = response_ids.shape[0]
    result_mask = torch.ones_like(response_ids, dtype=dtype)
    
    for i in range(batch_size):
        # Find positions of all EOS tokens in this sequence
        eos_positions = torch.nonzero(eos_mask[i], as_tuple=True)[0]
        if len(eos_positions) > 0:
            # Get the position of the last EOS token
            last_eos_pos = eos_positions[-1]
            # Set mask to 0 for all positions after the last EOS token
            result_mask[i, last_eos_pos+1:] = 0
    
    return result_mask

# def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, List], repeats: int, create_deep_copy: bool = False) -> Union[torch.Tensor, np.ndarray, List[Any]]:
#     print(f"_repeat_interleave value")
#     if isinstance(value, torch.Tensor):
#         return value.repeat_interleave(repeats, dim=0)
#     elif create_deep_copy:
#         print(f"_repeat_interleave create_deep_copy with {len(value)} items")
#         # print(value)
#         # For lists, create a new list with each element repeated 'repeats' times
#         result = []
#         for i, item in enumerate(value):
#             # Use deep copy to ensure nested structures are properly copied
#             try:
#                 for _ in range(repeats):
#                     result.append(copy.deepcopy(item))
#                 print(f"Successfully copied item {i}")
#             except Exception as e:
#                 print(f"Error deep copying item {i}: {e}")
#                 # Fallback to shallow copy if deep copy fails
#                 for _ in range(repeats):
#                     result.append(copy.copy(item))
#         return result
#     elif isinstance(value, np.ndarray):
#         return np.repeat(value, repeats, axis=0)
#     else:
#         # For other types, create a list with the value repeated
#         return [copy.deepcopy(value) for _ in range(repeats)]

def _parse_coordinate(text: str):
    """
    Extract a coordinate from text of the form '(x, y)'.
    Returns (x, y) as a tuple of ints if found, otherwise None.
    """
    try:
        # text = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        # print(f"_parse_coordinate text: {text}")
        text = text.strip()
        json_text = json.loads(text)
        return json_text["arguments"]["coordinate"]
    except Exception as e:
        # print(f"_parse_coordinate error: {e}")
        return None

def _get_point_crop(image: Image.Image, point: List[int], offset: int = 50) -> Image.Image:
    """
    Get a crop of the image centered on the point with a side length of 2*offset.
    Also draws a dot at the point location within the crop.
    """
    x, y = point
    width, height = image.size

    # Ensure crop boundaries are within image dimensions
    left = max(0, x - offset)
    top = max(0, y - offset)
    right = min(width, x + offset)
    bottom = min(height, y + offset)
    
    # Ensure that right > left and bottom > top
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    
    # Create the crop
    crop = image.crop((left, top, right, bottom))
    
    # Draw a dot at the point location (relative to the crop)
    draw = ImageDraw.Draw(crop)
    dot_x = x - left
    dot_y = y - top
    radius = 7
    draw.ellipse(
        [(dot_x - radius, dot_y - radius), (dot_x + radius, dot_y + radius)],
        fill="red",
        outline="white",
        width=2
    )

    # Super-resolution to 200x200
    crop = crop.resize((offset*2*2, offset*2*2), Image.Resampling.LANCZOS)

    return crop

def prepare_message_for_vllm(self, content_messages):
        """
        The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
        Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
        By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
        """
        vllm_messages, fps_list = [], []
        for message in content_messages:
            message_content_list = message["content"]
            if not isinstance(message_content_list, list):
                vllm_messages.append(message)
                continue

            new_content_list = []
            for part_message in message_content_list:
                if 'video' in part_message:
                    video_message = [{'content': [part_message]}]
                    image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                    assert video_inputs is not None, "video_inputs should not be None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get('fps', []))

                    # encode image with base64
                    base64_frames = []
                    for frame in video_input:
                        img = Image.fromarray(frame)
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        base64_frames.append(base64_str)

                    part_message = {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                    }
                new_content_list.append(part_message)
            message["content"] = new_content_list
            vllm_messages.append(message)
        return vllm_messages, {'fps': fps_list}

def time_to_seconds(ts: str):
    pattern = re.compile(r"""^(
        (?:(\d+):(\d{2})\.(\d+))$|                # MM:SS.ff
        (?:(\d{1,2}):(\d{2}):(\d{2})\.(\d+))$|    # HH:MM:SS.ff
        (?:(\d{1,2}):(\d{2}):(\d{2}))$            # HH:MM:SS / H:MM:SS
    )""", re.VERBOSE)

    m = pattern.match(ts)
    if not m:
        return None

    if m.group(2):  # MM:SS.ff
        mm, ss, ff = int(m.group(2)), int(m.group(3)), float("0."+m.group(4))
        return mm*60 + ss + ff
    elif m.group(5):  # HH:MM:SS.ff
        hh, mm, ss, ff = int(m.group(5)), int(m.group(6)), int(m.group(7)), float("0."+m.group(8))
        return hh*3600 + mm*60 + ss + ff
    else:  # HH:MM:SS / H:MM:SS
        hh, mm, ss = int(m.group(9)), int(m.group(10)), int(m.group(11))
        return hh*3600 + mm*60 + ss

def parse_times_json(text: str, total_duration: float) -> List[Tuple[str, str]]:
        """
        从字符串中提取 JSON 格式的起始时间和结束时间。
        JSON 格式示例：
        {"t_start":"mm:ss.ff","t_end":"mm:ss.ff","label":"..."}
        
        返回: [(start_second, end_second), ...]
        """
        times = []
        # 匹配 JSON 对象
        json_matches = re.findall(r'\{.*?\}', text)
        
        for j in json_matches:
            try:
                data = json.loads(j)
                if "start_time" in data and "end_time" in data:
                    st_second = time_to_seconds(data["start_time"])
                    ed_second = time_to_seconds(data["end_time"])
                    # TODO: 加上对起止时间有效性的判定（结束时间必须大于开始时间，都在视频总时长范围内）
                    if st_second and ed_second and ed_second > st_second and ed_second <= total_duration:
                        times.append((data["start_time"], data["end_time"], st_second, ed_second))
            except:
                try:
                    data = json.loads(json.loads(f'"{j}"'))
                    if "start_time" in data and "end_time" in data:
                        st_second = time_to_seconds(data["start_time"])
                        ed_second = time_to_seconds(data["end_time"])
                        # TODO: 加上对起止时间有效性的判定（结束时间必须大于开始时间，都在视频总时长范围内）
                        if st_second and ed_second and ed_second > st_second and ed_second <= total_duration:
                            times.append((data["start_time"], data["end_time"], st_second, ed_second))
                except json.JSONDecodeError:
                    continue  # 跳过无效 JSON

        return times


def restore_bboxes_in_text(text: str, orig_h: int, orig_w: int,
                           new_h: int, new_w: int) -> str:
    # 得到resize后的新尺寸
    # new_h, new_w = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels)

    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    pattern = re.compile(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")

    def replacer(match):
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = round(x1 * scale_x)
        x2 = round(x2 * scale_x)
        y1 = round(y1 * scale_y)
        y2 = round(y2 * scale_y)
        return f"[{x1}, {y1}, {x2}, {y2}]"

    return pattern.sub(replacer, text)


def extract_and_crop_bboxes(text: str, image: Image.Image) -> List[Image.Image]:
    """
    从文本中的 <tool_call> JSON 提取 bbox（支持单个或多个），
    并在原始 PIL 图像上裁剪对应区域。

    参数：
        text: 包含 <tool_call> 的完整文本。
        image: 原始 PIL.Image 图像。

    返回：
        List[Image.Image]: 裁剪得到的图像列表。
    """
    # 提取 <tool_call> 内容
    tool_call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not tool_call_match:
        return []

    tool_json_str = tool_call_match.group(1)

    try:
        tool_data = json.loads(tool_json_str)
    except json.JSONDecodeError:
        print(text)
        print("⚠️ JSON 解析失败")
        return []

    try:
        # 提取 bbox 信息
        bbox_data = tool_data.get("arguments", {}).get("bbox", None)
        if bbox_data is None:
            return []
        if not isinstance(bbox_data, list) or len(bbox_data) == 0:
            print(text)
            print("⚠️ bbox 列表长度为 0")
            return []

        # 如果是单个 bbox（1 维列表）
        if isinstance(bbox_data[0], (int, float)):
            bboxes = [bbox_data]
        # 如果是多个 bbox（2 维列表）
        elif isinstance(bbox_data[0], (list, tuple)) and isinstance(bbox_data[0][0], (int, float)):
            bboxes = bbox_data
        else:
            print(text)
            print("⚠️ bbox 格式不符合要求")
            return []
    except Exception as e:
        print(text)
        print(f"⚠️ json或bbox格式不符合要求，提取 bbox 时出错: {e}")
        return []

    cropped_images = []
    for bbox in bboxes:
        try:
            if len(bbox) != 4:
                continue  # 跳过不合法的 bbox
            x1, y1, x2, y2 = map(float, bbox)

            # 保证坐标合法
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)

            # 裁剪图像
            if x2 <= x1 or y2 <= y1:
                continue  # 跳过无效的 bbox
            cropped = image.crop((x1, y1, x2, y2))
            if cropped.width < 28 or cropped.height<28:
                resized_height, resized_width = smart_resize(
                    cropped.height,
                    cropped.width,
                    factor=28,
                )
                cropped = cropped.resize((resized_width, resized_height))
            cropped_images.append(cropped)
        except Exception as e:
            print(text)
            print(f"⚠️ 裁剪 bbox 时出错: {e}")
            continue

    return cropped_images

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)

class vLLMRolloutMultiturn_Traj(BaseRollout):
    def __init__(
        self, 
        model_path: str, 
        config: RolloutConfig, 
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin
        ):
        """A vLLM rollout that supports multi-turn conversation in batch.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        # if not config.enforce_eager and config.free_cache_engine:
        #     raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}
            self.limit_images = config.limit_images
        else:
            raise ValueError("limit_images should be greater than >>1.")

        token_seqs_to_mask = ["<|im_start|>", "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"]
        self.token_seqs_to_mask_ids = [self.tokenizer.encode(token_seq, add_special_tokens=False) for token_seq in token_seqs_to_mask]

        # self.inference_engine = LLM(
        #     model=model_path,
        #     skip_tokenizer_init=False,
        #     tensor_parallel_size=config.tensor_parallel_size,
        #     dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
        #     gpu_memory_utilization=config.gpu_memory_utilization,
        #     enforce_eager=config.enforce_eager,
        #     max_model_len=config.prompt_length + config.response_length,
        #     max_num_batched_tokens=config.max_num_batched_tokens,
        #     enable_sleep_mode=True,
        #     distributed_executor_backend="external_launcher",
        #     disable_custom_all_reduce=True,
        #     disable_mm_preprocessor_cache=True,
        #     disable_log_stats=config.disable_log_stats,
        #     enable_chunked_prefill=config.enable_chunked_prefill,
        #     seed=42,
        #     **vllm_init_kwargs,
        # )

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            # seed=config.seed,
            seed=int(os.getenv("RANK", "0")) // config.tensor_parallel_size,
            max_model_len=config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={"image": config.limit_images+50} if config.limit_images > 0 else None,
            disable_mm_preprocessor_cache=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        # breakpoint()

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.n = self.sampling_params.n
        self.sampling_params.n = 1

        self.max_iterations = config.max_iterations

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # breakpoint()
        # update sampling params
        old_sampling_params_args = {}
        old_n = self.n
        
        if kwargs:
            # Handle 'n' separately since it's stored in self.n
            if 'n' in kwargs:
                old_sampling_params_args['n'] = old_n
                self.n = kwargs['n']
            
            # Handle other sampling parameters
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

            self.sampling_params.n = 1
            # logger.info(f"Sampling params: {self.sampling_params}.")
            # logger.info(f"n: {self.n}.")

        yield
        
        # roll back to previous sampling params
        if 'n' in old_sampling_params_args:
            self.n = old_sampling_params_args.pop('n')
            
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

        self.sampling_params.n = 1
    
    def _get_response_loss_mask(
        self, 
        response_ids: List | torch.Tensor, 
        eos_token_id: int | List[int],
        additonal_token_ids_to_mask: List[List[int]] = [],
        dtype: torch.dtype = torch.int64,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Return a mask (same shape as `response_ids`) of dtype `dtype` which is `0`
        wherever `eos_token_id` occurs, or where any consecutive pattern in 
        `additonal_token_ids_to_mask` is found, else `1`.

        Args:
            response_ids: [batch_size, seq_len] token IDs (or possibly just [seq_len])
            eos_token_id: integer EOS token to mask, or list of integers
            additonal_token_ids_to_mask: list of lists, each list being a consecutive pattern
                                        that should be masked wherever found
            dtype: output tensor dtype

        Returns:
            mask of shape == response_ids.shape, dtype = `dtype`.
        """
        if isinstance(response_ids, list):
            response_ids = torch.tensor(response_ids)

        # Ensure 2D shape for uniform handling
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)  # [1, seq_len]

        bsz, seq_len = response_ids.shape
        mask = torch.ones_like(response_ids, dtype=dtype, device=device)

        # Mark wherever eos_token_id is found
        if isinstance(eos_token_id, list):
            for token_id in eos_token_id:
                mask[response_ids.eq(token_id)] = 0
        else:
            mask[response_ids.eq(eos_token_id)] = 0

        # For each row in the batch, mark occurrences of each consecutive pattern
        for b in range(bsz):
            row = response_ids[b]
            for pattern in additonal_token_ids_to_mask:
                pat_len = len(pattern)
                if pat_len == 0:
                    continue

                # Slide over this row and check for consecutive matches
                for start_idx in range(seq_len - pat_len + 1):
                    segment = row[start_idx : start_idx + pat_len]
                    # Compare this segment to the pattern
                    if torch.all(segment.eq(torch.tensor(pattern, device=row.device))):
                        mask[b, start_idx : start_idx + pat_len] = 0

        # If we artificially added a batch dimension, squeeze it out
        if response_ids.size(0) == 1 and bsz == 1:
            return mask.squeeze(0)
        return mask

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Multi-turn, batched conversation:
          - Each item in `prompts` is one conversation session.
          - We keep generating in "turns" for only those sessions that haven't output <answer>...</answer>.
          - If <tool>...</tool> is found, we parse its coordinate and append <observation> <image> </observation>.
          - End when each conversation has an <answer>.
        """

        print("HERE1")

        # For safety, have a max iteration to avoid infinite loops
        iteration_count = 0

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"] # [151645, 151643] -> ｜im_end｜, |end_of_text|
        input_prompt_generation_mask = torch.zeros_like(
            input_ids, dtype=attention_mask.dtype, device=attention_mask.device
        )  # (B'*R, max_prompt_length), all 0
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,  # if greedy, only 1 response
            }

        ##### Initialization #####
        vllm_inputs = (
            []
        )  # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [numpy array]
        multi_turn_response_mask = []  # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = []  # B*R, list of int, record first round prompt of all trajs
        crop_images = []


        # By default, we use the sampling_params we set up in the constructor.
        with self.update_sampling_params(**prompts.meta_info):

            print(f"Sampling params: {self.sampling_params}.")
            print(f"n: {self.n}.")

            print("HERE2")
            print(f"-----------------self.n: {self.n}-----------------")

            if self.n > 1:
                batch_size = batch_size * self.n
                input_ids = _repeat_interleave(input_ids, self.n)
                attention_mask = _repeat_interleave(attention_mask, self.n)
                position_ids = _repeat_interleave(position_ids, self.n)
                raw_prompt_ids = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.n
                        )
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    print("HERE2.1")
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.n
                        )
                    input_images = []
                    input_image_paths = []
                    for i, image_data in enumerate(non_tensor_batch["multi_modal_data"]):
                        for _ in range(self.n):
                            input_images.append([im.copy() for im in image_data["image"]])
                            input_image_paths.append(image_data['image_path'])
                    print("HERE2.4")
            else:
                raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    input_images = []
                    input_image_paths = []
                    for i, image_data in enumerate(non_tensor_batch["multi_modal_data"]):
                        input_images.append([im.copy() for im in image_data["image"]])
                        input_image_paths.append(image_data['image_path'])
                    print("HERE2.4")

            if 'multi_modal_data' in non_tensor_batch:
                _multi_modal_data_list = non_tensor_batch['multi_modal_data']
                for raw_prompt_ids_, multi_modal_data in zip(non_tensor_batch['raw_prompt_ids'], _multi_modal_data_list):
                    prefix_length = len(raw_prompt_ids_)
                    for _ in range(self.n):
                        multi_turn_response_mask.append(
                            [
                                torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                            ]  # USER, Mark as 0
                        )  # [torch.Tensor(prefix_length,)]
                        prefix_prompt_lengths.append(prefix_length)
                        crop_images.append([])  # init as empty lists
            non_tensor_batch["raw_prompt_ids"] = raw_prompt_ids.copy()       

            print("HERE3")

            ##### Loop Setting #####
            is_finished = [False]*batch_size

            if batch_size != len(raw_prompt_ids):
                print('------------------------------------------------------------------------------')
                print(f'batch_size:  {batch_size}')
                print(f'raw_prompt_ids length:  {len(raw_prompt_ids)}')
                print('------------------------------------------------------------------------------')
                raise RuntimeError("vllm sharding manager is not work properly.")


            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids_, multi_modal_data in zip(
                    raw_prompt_ids, input_images
                ):
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids_), "multi_modal_data": {"image": multi_modal_data}})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": list(raw_prompt_ids_)} for raw_prompt_ids_ in raw_prompt_ids
                ]

            response_ids = [[] for _ in range(batch_size)]

            response_loss_mask = [torch.tensor([], dtype=attention_mask.dtype, device=attention_mask.device) for _ in range(batch_size)]

            print("HERE4")
            # breakpoint()
        
            while not all(is_finished) and (iteration_count < self.max_iterations):
                print('------------------------------------------------------------')
                print(f'iteration_count:  {iteration_count}')
                print('------------------------------------------------------------')
                iteration_count += 1

                # Build prompts for only the *active* items
                active_indices = [i for i, done in enumerate(is_finished) if not done]
                if not active_indices:
                    break

                # For each active item, we pass the entire conversation so far
                batch_prompts = []
                for i in active_indices:
                    batch_prompts.append(vllm_inputs[i])

                print("HERE5")

                # Generate in *one batch* for all active conversations
                try:

                    generation_results: List[RequestOutput] = self.inference_engine.generate(
                        prompts=batch_prompts, 
                        sampling_params=self.sampling_params, 
                        use_tqdm=(self.rank == 0)
                    )
                except:
                    breakpoint()
                    
                    print('--------------------------------generate failure---------------------------------')
                    ##### re-build response #####
                    response = []  # B'*R, torch.Tensors with unequal lengths
                    response_generation_mask = []  # B'*R, torch.Tensors with unequal lengths but align with 'response'
                    max_input_id_length = 0
                    # process search tool returned images
                    for i_ in range(batch_size):  # bs*n
                        # for each traj, we skip first-round prompt_ids/attention_mask
                        all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
                        resp_mask_device = all_response_masks.device

                        first_round_prompt_length = prefix_prompt_lengths[i_]
                        response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

                        # NOTE: [For Multi-Image] Update response_after_prompt(list of token_ids) and all_response_masks if search tool returned images
                        if crop_images[i_]:
                            # process PIL.Images to get 'pixel_values' and 'image_grid_thw'
                            cropped_image_inputs = self.processor.image_processor(
                                images=crop_images[i_], videos=None, return_tensors='pt'
                            )  # dict_keys(['pixel_values_videos', 'video_grid_thw'])
                            cropped_image_grid_thw = cropped_image_inputs['image_grid_thw']
                            # print(f"searched_image_grid_thw shape: {searched_image_grid_thw.shape}")
                            # print(f"searched_image_grid_thw: {searched_image_grid_thw}")
                            if cropped_image_grid_thw is not None:
                                merge_length = self.processor.image_processor.merge_size**2
                                index, image_pad_token, magic_num = 0, 151655, 654321
                                all_response_masks = all_response_masks.tolist()  # for convenient modification
                                while image_pad_token in response_after_prompt:
                                    # find pos of <|image_pad|>
                                    pos = response_after_prompt.index(image_pad_token)
                                    replicate_count = cropped_image_grid_thw[index].prod() // merge_length
                                    # update response_after_prompt
                                    response_after_prompt[pos : pos + 1] = [magic_num] * replicate_count
                                    # update all_response_masks
                                    all_response_masks[pos : pos + 1] = [0] * replicate_count
                                    index += 1
                                response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                                all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)

                        if len(response_after_prompt) + prefix_prompt_lengths[i_] > max_input_id_length:
                            max_input_id_length = len(response_after_prompt) + prefix_prompt_lengths[i_]

                        response_generation_mask.append(all_response_masks)  # at least we have single-turn conversation
                        all_response = torch.tensor(response_after_prompt, device=input_ids.device, dtype=input_ids.dtype)
                        response.append(all_response)

                    print(f'iteration_count:  {iteration_count},   max_input_id_length:   {max_input_id_length}')
                    print('--------------------------------generate failure---------------------------------')
                    
                    generation_results: List[RequestOutput] = self.inference_engine.generate(
                        prompts=batch_prompts, 
                        sampling_params=self.sampling_params, 
                        use_tqdm=(self.rank == 0)
                    )




                if self.rank == 0:
                    print("HERE6")

                # breakpoint()
                # Now parse results: each item in generation_results corresponds to one prompt in `batch_prompts`.
                for idx, output in tqdm(enumerate(generation_results), 
                                       desc="Processing generation results", 
                                       leave=False,
                                       disable=self.rank != 0):
                    i = active_indices[idx]  # index in the original batch
                    if is_finished[i]:
                        continue  # skip if finished

                    if not output.outputs:
                        # No model output?
                        is_finished[i] = True
                        continue

                    tokens = list(output.outputs[0].token_ids) # zsr: 检查一下output.outputs的维度
                    filtered_token_ids = [token_id for token_id in tokens if token_id <= 151664]
                    if len(filtered_token_ids) > self.config.max_generation_length_per_turn:
                        filtered_token_ids = filtered_token_ids[:self.config.max_generation_length_per_turn]
                    if 151645 not in filtered_token_ids:
                        # replace the last token with <|im_end|> if no <|im_end|> in response,
                        # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                        filtered_token_ids[-1] = 151645

                    # Append to the conversation
                    vllm_inputs[i]["prompt_token_ids"] += filtered_token_ids

                    response_ids[i] += filtered_token_ids

                    response_loss_mask_ = torch.ones(len(filtered_token_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                    response_loss_mask_ = response_loss_mask_ * self._get_response_loss_mask(filtered_token_ids, eos_token_id, self.token_seqs_to_mask_ids, dtype=attention_mask.dtype, device=attention_mask.device)

                    response_loss_mask[i] = torch.cat([response_loss_mask[i], response_loss_mask_], dim=-1)

                    multi_turn_response_mask[i].append(
                        torch.ones(len(filtered_token_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                    )  # ASSISTANT, Mark as 1

                    cur_text = self.tokenizer.decode(filtered_token_ids)

                    # print("HERE7")

                    # Check for <answer>...</answer>
                    # answer_match = re.search(r"<answer>(.*?)</answer>", cur_text, re.DOTALL)
                    answer_match = re.search(r"(.*?)</answer>", cur_text, re.DOTALL)
                    if answer_match:
                        # Mark done
                        is_finished[i] = True
                        continue

                    if len(vllm_inputs[i]["multi_modal_data"]["image"]) >= self.limit_images:
                        is_finished[i] = True
                        continue

                    # Check & Parse Crop Operation
                    # breakpoint()
                    cropped_images_ = extract_and_crop_bboxes(cur_text, input_images[i][0])
                    if cropped_images_:
                        # breakpoint()
                        # For each found <tool>, parse coordinate, crop, and append <observation> <image> </observation>
                        if vllm_inputs[i]["multi_modal_data"] is not None:

                            if len(vllm_inputs[i]["multi_modal_data"]["image"]) >= self.limit_images - 1:
                                # breakpoint()
                                if len(cropped_images_) == 1:
                                    obs_prompt = "<observation>\nHere is the crop of the image showing the region:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation><|im_end|>\n<|im_start|>assistant\n<think>\nBased on all the regions I've examined, I can now provide my final answer.\n</think>\n<answer>"
                                else:
                                    obs_prompt = f"<observation>\nHere are the crops of the image showing {len(cropped_images_)} regions:\n"+f"<|vision_start|><|image_pad|><|vision_end|>\n"*len(cropped_images_)+"</observation><|im_end|>\n<|im_start|>assistant\n<think>\nBased on all the regions I've examined, I can now provide my final answer.\n</think>\n<answer>"
                            else:
                                if len(cropped_images_) == 1:
                                    obs_prompt = "<observation>\nHere is the crop of the image showing the region:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation><|im_end|>\n<|im_start|>assistant\n"
                                else:
                                    obs_prompt = f"<observation>\nHere are the crops of the image showing {len(cropped_images_)} regions:\n"+f"<|vision_start|><|image_pad|><|vision_end|>\n"*len(cropped_images_)+"</observation><|im_end|>\n<|im_start|>assistant\n"

                            vllm_inputs[i]["multi_modal_data"]["image"].extend(cropped_images_)
                            crop_images[i].extend(cropped_images_)
                            
                            obs_inputs = self.processor(
                                images=cropped_images_, 
                                text=[obs_prompt],
                                add_special_tokens=False,
                                return_tensors="pt"
                            )

                            obs_input_ids = obs_inputs["input_ids"][0].cpu().tolist()
                            obs_raw_prompt_ids = self.tokenizer.encode(obs_prompt, add_special_tokens=False)

                            cur_text = self.tokenizer.decode(obs_input_ids)

                            vllm_inputs[i]["prompt_token_ids"].extend(obs_raw_prompt_ids)
                            response_ids[i].extend(obs_raw_prompt_ids)

                            response_loss_mask_ = torch.zeros(len(obs_raw_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                            multi_turn_response_mask[i].append(
                                torch.zeros(len(obs_raw_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                            )  # USER, Mark as 0

                            if len(vllm_inputs[i]["multi_modal_data"]["image"]) >= self.limit_images:
                                num_to_unmask = len(self.tokenizer.encode("<answer>", add_special_tokens=False))
                                response_loss_mask_[-num_to_unmask:] = torch.ones(num_to_unmask, dtype=attention_mask.dtype, device=attention_mask.device)

                            response_loss_mask[i] = torch.cat([response_loss_mask[i], response_loss_mask_], dim=-1)

                            # print("HERE8")
                            # logger.info("HEREHERE")
                            # print("HERE8")
                    if iteration_count == self.max_iterations - 1:
                        # breakpoint()
                        end_prompt = f"\n<|im_start|>assistant\n<think>\nBased on all the information I've gathered, I'll now provide my final answer.\n</think>\n<answer>"
                        end_prompt_ids = self.tokenizer.encode(end_prompt, add_special_tokens=False)
                        vllm_inputs[i]["prompt_token_ids"].extend(end_prompt_ids)
                        response_ids[i].extend(end_prompt_ids)
                        multi_turn_response_mask[i].append(
                            torch.ones(len(end_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                        )  # ASSISTANT, Mark as 1

        if self.rank == 0:
            print("HERE7")
        logger.info("HERE7")

        ##### re-build response #####
        response = []  # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = []  # B'*R, torch.Tensors with unequal lengths but align with 'response'
        # process search tool returned images
        for i_ in range(batch_size):  # bs*n
            # for each traj, we skip first-round prompt_ids/attention_mask
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            # NOTE: [For Multi-Image] Update response_after_prompt(list of token_ids) and all_response_masks if search tool returned images
            if crop_images[i_]:
                # process PIL.Images to get 'pixel_values' and 'image_grid_thw'
                cropped_image_inputs = self.processor.image_processor(
                    images=crop_images[i_], videos=None, return_tensors='pt'
                )  # dict_keys(['pixel_values_videos', 'video_grid_thw'])
                cropped_image_grid_thw = cropped_image_inputs['image_grid_thw']
                # print(f"searched_image_grid_thw shape: {searched_image_grid_thw.shape}")
                # print(f"searched_image_grid_thw: {searched_image_grid_thw}")
                if cropped_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index, image_pad_token, magic_num = 0, 151655, 654321
                    all_response_masks = all_response_masks.tolist()  # for convenient modification
                    while image_pad_token in response_after_prompt:
                        # find pos of <|image_pad|>
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = cropped_image_grid_thw[index].prod() // merge_length
                        # update response_after_prompt
                        response_after_prompt[pos : pos + 1] = [magic_num] * replicate_count
                        # update all_response_masks
                        all_response_masks[pos : pos + 1] = [0] * replicate_count
                        index += 1
                    response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                    all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)

            response_generation_mask.append(all_response_masks)  # at least we have single-turn conversation
            all_response = torch.tensor(response_after_prompt, device=input_ids.device, dtype=input_ids.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), f"shape mismatched | response[i_]: {response[i_].shape[0]} | response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(
            response_generation_mask
        ), "length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(
            response, self.pad_token_id, dim=0
        )  # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0)  # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.response_length)
        if response.shape[1] > self.config.response_length:
            response = response[:, : self.config.response_length]
            response_generation_mask = response_generation_mask[:, : self.config.response_length]
        elif response.shape[1] < self.config.response_length:
            response = VF.pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_generation_mask = VF.pad_sequence_to_length(
                response_generation_mask, self.config.response_length, 0
            )


        # All for 1st USER prompt
        if self.n > 1 and do_sample:
            # NOTE: We repeat 'multi_modal_data'
            if 'multi_modal_data' in non_tensor_batch.keys():
                repeated = []
                _index_br = 0
                for item in non_tensor_batch['multi_modal_data']:
                    for _ in range(self.n):
                        new_item = copy.deepcopy(item)
                        if crop_images[_index_br]:
                            new_item['image'] += crop_images[_index_br]
                        repeated.append(new_item)
                        _index_br += 1
                non_tensor_batch['multi_modal_data'] = repeated
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, self.n
            )  # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        # NOTE: transform 'multi_modal_data' to 'multi_modal_inputs'
        _processed_images = [self.processor.image_processor(
            images=_multi_modal_data['image'],
            videos=None,
            return_tensors="pt"
        ) for _multi_modal_data in non_tensor_batch['multi_modal_data']]
        _processed_images = [{key: val for key, val in image_inputs.items() if key in ['pixel_values', 'image_grid_thw']} for image_inputs in _processed_images]
        _multi_modal_inputs = np.array(_processed_images, dtype=object)
        non_tensor_batch['multi_modal_inputs'] = _multi_modal_inputs

        seq = torch.cat([input_ids, response], dim=-1)  # (B*R, max_prompt_length+max_response_length_total)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)


        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # breakpoint()
        response_position_ids = position_ids[..., -1:] + delta_position_id # zsr: 为什么不用get_rope_index(), 而就直接像1d那样加上了？
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask_last(
            response_ids=response, eos_token_id=[151645], dtype=attention_mask.dtype
        )  # HACK: for qwen, |im_end| is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': input_ids.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'response_loss_mask': multi_turn_response_mask.contiguous(),
                'response_mask': response_generation_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

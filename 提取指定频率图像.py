"""
频移图像提取工具 (改进版)

功能：
- 从频移图像文件夹中提取指定频率范围的四步相移图像
- 支持不同的频率步长（1, 0.5, 2等）
- 自动识别水平/垂直条纹模式
- 可选的图像重命名功能
- 多线程支持提高复制速度
- 配置参数支持直接修改运行
- 创建新的文件夹并按规范命名

使用场景：
- FSM解码中需要特定频率范围的图像
- 批量处理频移图像数据

作者：基于FSM解码需求创建
日期：2025-11-14
更新：2025-11-14 - 添加多线程和配置参数支持
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# 导入重命名工具
from rename_images import ImageRenamer, rename_images_simple



# 线程安全的进度计数器
class ProgressCounter:
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.lock = threading.Lock()
    
    def increment(self, show_progress: bool = True):
        with self.lock:
            self.current += 1
            if show_progress and (self.current % 10 == 0 or self.current == self.total):
                print(f"\r  进度: {self.current}/{self.total}", end="")


class FrequencyImageExtractor:
    """
    频移图像提取工具类
    """
    
    def __init__(self, source_dir: str):
        """
        初始化频移图像提取器
        
        Parameters:
        -----------
        source_dir : str
            源图像文件夹路径
        """
        self.source_dir = Path(source_dir)
        
        if not self.source_dir.exists():
            raise ValueError(f"源文件夹不存在: {source_dir}")
        
        if not self.source_dir.is_dir():
            raise ValueError(f"路径不是文件夹: {source_dir}")
    
    def detect_pattern_type(self) -> str:
        """
        从文件夹名称检测条纹模式（水平/垂直）
        
        Returns:
        --------
        str
            "horizontal" 或 "vertical"
        """
        folder_name = self.source_dir.name.lower()
        
        # 检测关键词
        if any(keyword in folder_name for keyword in ['horizontal', 'hor', 'h_', '_h_', 'hz']):
            return "horizontal"
        elif any(keyword in folder_name for keyword in ['vertical', 'ver', 'v_', '_v_', 'vt']):
            return "vertical"
        else:
            # 默认根据常见命名模式判断
            if 'ver' in folder_name:
                return "vertical"
            elif 'hor' in folder_name:
                return "horizontal"
            else:
                # 如果无法判断，询问用户
                print(f"⚠ 无法从文件夹名称 '{self.source_dir.name}' 自动识别条纹模式")
                print("请手动选择:")
                print("1. horizontal (水平条纹)")
                print("2. vertical (垂直条纹)")
                choice = input("请选择 (1-2): ").strip()
                
                if choice == "2":
                    return "vertical"
                else:
                    return "horizontal"
    
    def preprocess_images(self) -> bool:
        """
        预处理图像：重命名为标准格式 (0.png, 1.png, ...)
        
        Returns:
        --------
        bool
            是否成功完成预处理
        """
        print("="*80)
        print("步骤1: 预处理图像文件")
        print("="*80)
        print(f"正在重命名文件夹中的图像: {self.source_dir}")
        
        try:
            # 使用现有的重命名工具
            success_count = rename_images_simple(
                str(self.source_dir), 
                output_format="png", 
                preview_only=False
            )
            
            if success_count > 0:
                print(f"✓ 成功重命名 {success_count} 个图像文件")
                return True
            else:
                print("❌ 没有找到需要重命名的图像文件")
                return False
                
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            return False
    
    def get_frequency_images(self, freq_start: float, freq_end: float, 
                           freq_step: float = 1.0) -> List[List[Path]]:
        """
        获取指定频率范围内的四步相移图像
        
        Parameters:
        -----------
        freq_start : float
            起始频率
        freq_end : float
            结束频率
        freq_step : float
            频率步长
            
        Returns:
        --------
        List[List[Path]]
            每个频率对应的四步相移图像路径列表
        """
        frequency_groups = []
        
        # 计算频率点
        current_freq = freq_start
        while current_freq <= freq_end:
            # 每个频率对应4张相移图像
            freq_images = []
            base_index = int(current_freq * (1.0 / freq_step) * 4)
            
            for phase in range(4):
                image_index = base_index + phase
                image_path = self.source_dir / f"{image_index}.png"
                
                if image_path.exists():
                    freq_images.append(image_path)
                else:
                    print(f"⚠ 警告: 图像不存在 {image_path}")
            
            if len(freq_images) == 4:
                frequency_groups.append(freq_images)
            else:
                print(f"⚠ 警告: 频率 {current_freq} 的图像不完整 (找到 {len(freq_images)}/4 张)")
            
            # 更新频率（处理浮点数精度问题）
            current_freq = round(current_freq + freq_step, 1)
        
        return frequency_groups
    
    def create_output_folder(self, freq_start: float, freq_end: float, 
                           pattern_type: str) -> Path:
        """
        创建输出文件夹
        
        Parameters:
        -----------
        freq_start : float
            起始频率
        freq_end : float
            结束频率
        pattern_type : str
            条纹模式 ("horizontal" 或 "vertical")
            
        Returns:
        --------
        Path
            输出文件夹路径
        """
        # 格式化频率（去除不必要的小数点）
        start_str = f"f{int(freq_start)}" if freq_start.is_integer() else f"f{freq_start}"
        end_str = f"f{int(freq_end)}" if freq_end.is_integer() else f"f{freq_end}"
        
        folder_name = f"FSM_{pattern_type}_{start_str}_{end_str}"
        output_dir = self.source_dir.parent / folder_name
        
        # 如果文件夹已存在，询问是否覆盖
        if output_dir.exists():
            print(f"⚠ 输出文件夹已存在: {output_dir}")
            choice = input("是否覆盖? (yes/no): ").strip().lower()
            if choice in ['yes', 'y', '是']:
                shutil.rmtree(output_dir)
            else:
                print("❌ 操作已取消")
                return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def copy_single_image(self, source_path: Path, target_path: Path, 
                         progress_counter: ProgressCounter, show_progress: bool = True) -> bool:
        """
        复制单个图像文件（线程安全）
        
        Parameters:
        -----------
        source_path : Path
            源文件路径
        target_path : Path
            目标文件路径
        progress_counter : ProgressCounter
            进度计数器
            
        Returns:
        --------
        bool
            是否成功
        """
        try:
            shutil.copy2(source_path, target_path)
            progress_counter.increment(show_progress)
            return True
        except Exception as e:
            print(f"\n❌ 复制失败 {source_path} -> {target_path}: {e}")
            return False
    
    def copy_images_multithreaded(self, frequency_groups: List[List[Path]], 
                                 output_dir: Path, freq_start: float, 
                                 freq_step: float, max_workers: int = 8, 
                                 show_progress: bool = True) -> int:
        """
        多线程复制图像文件
        
        Parameters:
        -----------
        frequency_groups : List[List[Path]]
            频率图像组列表
        output_dir : Path
            输出文件夹
        freq_start : float
            起始频率
        freq_step : float
            频率步长
        max_workers : int
            最大线程数
            
        Returns:
        --------
        int
            成功复制的文件数量
        """
        # 准备复制任务
        copy_tasks = []
        for freq_idx, freq_images in enumerate(frequency_groups):
            for phase_idx, image_path in enumerate(freq_images):
                new_index = freq_idx * 4 + phase_idx
                new_name = f"{new_index}.png"
                new_path = output_dir / new_name
                copy_tasks.append((image_path, new_path))
        
        total_files = len(copy_tasks)
        progress_counter = ProgressCounter(total_files)
        success_count = 0
        
        print(f"使用 {max_workers} 个线程复制 {total_files} 个文件...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.copy_single_image, source, target, progress_counter, show_progress): (source, target)
                for source, target in copy_tasks
            }
            
            # 等待任务完成
            for future in as_completed(future_to_task):
                if future.result():
                    success_count += 1
        
        print()  # 换行
        return success_count
    
    def copy_images_sequential(self, frequency_groups: List[List[Path]], 
                              output_dir: Path, freq_start: float, 
                              freq_step: float, show_progress: bool = True) -> int:
        """
        顺序复制图像文件
        
        Parameters:
        -----------
        frequency_groups : List[List[Path]]
            频率图像组列表
        output_dir : Path
            输出文件夹
        freq_start : float
            起始频率
        freq_step : float
            频率步长
            
        Returns:
        --------
        int
            成功复制的文件数量
        """
        total_copied = 0
        for freq_idx, freq_images in enumerate(frequency_groups):
            for phase_idx, image_path in enumerate(freq_images):
                new_index = freq_idx * 4 + phase_idx
                new_name = f"{new_index}.png"
                new_path = output_dir / new_name
                
                try:
                    shutil.copy2(image_path, new_path)
                    total_copied += 1
                except Exception as e:
                    print(f"❌ 复制失败 {image_path} -> {new_path}: {e}")
            
            # 显示进度
            if show_progress and ((freq_idx + 1) % 10 == 0 or freq_idx == len(frequency_groups) - 1):
                print(f"\r  进度: {freq_idx + 1}/{len(frequency_groups)} 个频率", end="")
        
        print()  # 换行
        return total_copied
    
    def extract_frequency_range(self, freq_start: float, freq_end: float, 
                              freq_step: float = 1.0, 
                              auto_preprocess: bool = True,
                              use_multithreading: bool = True,
                              max_workers: int = 8,
                              show_progress: bool = True) -> bool:
        """
        提取指定频率范围的图像
        
        Parameters:
        -----------
        freq_start : float
            起始频率
        freq_end : float
            结束频率
        freq_step : float
            频率步长
        auto_preprocess : bool
            是否自动预处理图像
            
        Returns:
        --------
        bool
            是否成功完成提取
        """
        print("="*80)
        print("频移图像提取工具")
        print("="*80)
        print(f"源文件夹: {self.source_dir}")
        print(f"频率范围: {freq_start} - {freq_end}")
        print(f"频率步长: {freq_step}")
        print("="*80)
        
        # 预处理图像
        if auto_preprocess:
            if not self.preprocess_images():
                return False
        
        # 检测条纹模式
        pattern_type = self.detect_pattern_type()
        print(f"\n检测到条纹模式: {pattern_type}")
        
        # 创建输出文件夹
        output_dir = self.create_output_folder(freq_start, freq_end, pattern_type)
        if output_dir is None:
            return False
        
        print(f"\n输出文件夹: {output_dir}")
        
        # 获取频率图像
        print("="*80)
        print("步骤2: 提取频率图像")
        print("="*80)
        
        frequency_groups = self.get_frequency_images(freq_start, freq_end, freq_step)
        
        if not frequency_groups:
            print("❌ 没有找到符合条件的频率图像")
            return False
        
        print(f"找到 {len(frequency_groups)} 个频率的图像组")
        
        # 复制图像到输出文件夹
        print("\n步骤3: 复制图像文件")
        print("="*80)
        
        if use_multithreading:
            total_copied = self.copy_images_multithreaded(
                frequency_groups, output_dir, freq_start, freq_step, max_workers, show_progress
            )
        else:
            total_copied = self.copy_images_sequential(
                frequency_groups, output_dir, freq_start, freq_step, show_progress
            )
        
        # 完成
        print("\n" + "="*80)
        print("✓ 提取完成！")
        print("="*80)
        print(f"成功复制: {total_copied} 个图像文件")
        print(f"频率数量: {len(frequency_groups)} 个")
        print(f"输出文件夹: {output_dir}")
        
        return True


if __name__ == "__main__":
    # ==================== 配置参数区域 ====================
    # 直接修改这些参数，然后运行脚本即可
    
    # 源图像文件夹路径
    SOURCE_DIRECTORY = r"F:\Projector_images\images_FSM\01_bookshelf_blue_cycle1\FSM_horizontal_f0_f150"
    
    # 频率范围设置
    FREQ_START = 0.0      # 起始频率
    FREQ_END = 100.0      # 结束频率
    FREQ_STEP = 1.0       # 频率步长
    
    # 是否自动预处理（重命名图像）
    AUTO_PREPROCESS = False
    
    # 多线程设置
    MAX_WORKERS = 8       # 最大线程数
    USE_MULTITHREADING = True  # 是否使用多线程
    
    # 输出设置
    SHOW_PROGRESS = True  # 是否显示进度
    
    # ==================== 配置参数区域结束 ====================
    
    print("="*80)
    print("频移图像提取工具")
    print("="*80)
    print(f"源目录: {SOURCE_DIRECTORY}")
    print(f"频率范围: {FREQ_START} - {FREQ_END}")
    print(f"频率步长: {FREQ_STEP}")
    print(f"自动预处理: {AUTO_PREPROCESS}")
    print(f"多线程: {USE_MULTITHREADING}")
    if USE_MULTITHREADING:
        print(f"线程数: {MAX_WORKERS}")
    print(f"显示进度: {SHOW_PROGRESS}")
    print("="*80)
    
    if not Path(SOURCE_DIRECTORY).exists():
        print(f"❌ 错误: 源目录不存在: {SOURCE_DIRECTORY}")
        exit(1)
    
    try:
        extractor = FrequencyImageExtractor(SOURCE_DIRECTORY)
        success = extractor.extract_frequency_range(
            freq_start=FREQ_START,
            freq_end=FREQ_END,
            freq_step=FREQ_STEP,
            auto_preprocess=AUTO_PREPROCESS,
            use_multithreading=USE_MULTITHREADING,
            max_workers=MAX_WORKERS,
            show_progress=SHOW_PROGRESS
        )
        
        if success:
            print("\n🎉 提取完成！")
        else:
            print("\n❌ 提取失败")
            exit(1)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

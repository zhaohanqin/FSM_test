"""
图像分类整理工具
用于将图像按照不同规则进行分组和整理

================================================================================
功能说明
================================================================================

功能1：【按相位模式分类】
    将图像按 4*i+1, 4*i+2, 4*i+3, 4*i+4 的模式分类
    - 第1,5,9,13...张图像 → phase0 文件夹 (4*i+1, i=0,1,2,...)
    - 第2,6,10,14...张图像 → phase1 文件夹 (4*i+2, i=0,1,2,...)
    - 第3,7,11,15...张图像 → phase2 文件夹 (4*i+3, i=0,1,2,...)
    - 第4,8,12,16...张图像 → phase3 文件夹 (4*i+4, i=0,1,2,...)

功能2：【按频率组打包】
    每4张连续图像打包成一个文件夹
    - 第1-4张图像   → f0 文件夹
    - 第5-8张图像   → f1 文件夹
    - 第9-12张图像  → f2 文件夹
    - ...

================================================================================
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import re

def natural_sort_key(filename: str) -> List:
    """
    自然排序的键函数，用于正确排序包含数字的文件名
    例如：img1.bmp, img2.bmp, ..., img10.bmp, img11.bmp
    而不是：img1.bmp, img10.bmp, img11.bmp, img2.bmp
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', filename)]

def get_sorted_images(input_dir: str, extensions: Tuple[str, ...] = ('.bmp', '.png', '.jpg', '.jpeg')) -> List[str]:
    """
    获取目录中的所有图像文件并按自然顺序排序
    
    Parameters:
        input_dir: 输入目录路径
        extensions: 支持的图像扩展名
    
    Returns:
        排序后的图像文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 获取所有图像文件
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # 按文件名自然排序
    image_files = sorted(image_files, key=lambda x: natural_sort_key(x.name))
    
    return [str(f) for f in image_files]

def classify_by_phase_pattern(input_dir: str, output_dir: str, copy_mode: bool = True):
    """
    功能1：按相位模式分类
    将图像按 4*i+k 模式分类到不同文件夹
    
    Parameters:
        input_dir: 输入目录（包含所有图像）
        output_dir: 输出目录（将创建 phase0, phase1, phase2, phase3 子文件夹）
        copy_mode: True=复制文件，False=移动文件
    """
    print("\n" + "=" * 70)
    print("【功能1：按相位模式分类】")
    print("=" * 70)
    
    # 获取排序后的图像列表
    images = get_sorted_images(input_dir)
    total_images = len(images)
    
    if total_images == 0:
        print(f"❌ 未在目录中找到图像文件: {input_dir}")
        return
    
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🖼️  找到图像: {total_images} 张")
    
    # 检查图像数量是否为4的倍数
    if total_images % 4 != 0:
        print(f"⚠️  警告：图像总数 ({total_images}) 不是4的倍数！")
        print(f"    最后 {total_images % 4} 张图像将按相同规则处理")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建4个相位文件夹
    phase_folders = []
    for phase in range(4):
        phase_folder = output_path / f"phase{phase}"
        phase_folder.mkdir(exist_ok=True)
        phase_folders.append(phase_folder)
    
    # 分类处理
    print("\n开始分类处理...")
    print(f"模式: {'复制' if copy_mode else '移动'} 文件")
    print("-" * 70)
    
    for idx, img_path in enumerate(images, start=1):
        # 计算相位索引：第1,5,9... → phase0; 第2,6,10... → phase1; ...
        phase_idx = (idx - 1) % 4
        
        # 目标路径
        img_name = Path(img_path).name
        dest_path = phase_folders[phase_idx] / img_name
        
        # 复制或移动文件
        if copy_mode:
            shutil.copy2(img_path, dest_path)
            action = "复制"
        else:
            shutil.move(img_path, dest_path)
            action = "移动"
        
        print(f"  [{idx:4d}/{total_images}] {action}: {img_name:30s} → phase{phase_idx}/")
    
    # 统计信息
    print("-" * 70)
    print("✅ 分类完成！")
    print("\n📊 统计信息:")
    for phase in range(4):
        count = len(list(phase_folders[phase].glob('*')))
        print(f"  phase{phase}/  {count} 张图像")
    print("=" * 70)

def group_by_frequency(input_dir: str, output_dir: str, copy_mode: bool = True, 
                       images_per_group: int = 4, folder_prefix: str = "f"):
    """
    功能2：按频率组打包
    每N张连续图像打包成一个文件夹
    
    Parameters:
        input_dir: 输入目录（包含所有图像）
        output_dir: 输出目录（将创建 f0, f1, f2, ... 子文件夹）
        copy_mode: True=复制文件，False=移动文件
        images_per_group: 每组图像数量（默认4）
        folder_prefix: 文件夹前缀（默认"f"）
    """
    print("\n" + "=" * 70)
    print("【功能2：按频率组打包】")
    print("=" * 70)
    
    # 获取排序后的图像列表
    images = get_sorted_images(input_dir)
    total_images = len(images)
    
    if total_images == 0:
        print(f"❌ 未在目录中找到图像文件: {input_dir}")
        return
    
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🖼️  找到图像: {total_images} 张")
    print(f"📦 每组图像: {images_per_group} 张")
    
    # 计算组数
    total_groups = (total_images + images_per_group - 1) // images_per_group
    print(f"📂 将创建文件夹: {total_groups} 个 ({folder_prefix}0 ~ {folder_prefix}{total_groups-1})")
    
    # 检查是否有剩余
    remainder = total_images % images_per_group
    if remainder != 0:
        print(f"⚠️  警告：最后一组只有 {remainder} 张图像（不足{images_per_group}张）")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 分组处理
    print("\n开始分组处理...")
    print(f"模式: {'复制' if copy_mode else '移动'} 文件")
    print("-" * 70)
    
    for idx, img_path in enumerate(images):
        # 计算组索引（从0开始）
        group_idx = idx // images_per_group
        
        # 创建组文件夹
        group_folder = output_path / f"{folder_prefix}{group_idx}"
        group_folder.mkdir(exist_ok=True)
        
        # 目标路径
        img_name = Path(img_path).name
        dest_path = group_folder / img_name
        
        # 复制或移动文件
        if copy_mode:
            shutil.copy2(img_path, dest_path)
            action = "复制"
        else:
            shutil.move(img_path, dest_path)
            action = "移动"
        
        # 在组内的索引
        in_group_idx = idx % images_per_group
        print(f"  [{idx+1:4d}/{total_images}] {action}: {img_name:30s} → {folder_prefix}{group_idx}/ (第{in_group_idx+1}张)")
    
    # 统计信息
    print("-" * 70)
    print("✅ 分组完成！")
    print("\n📊 统计信息:")
    for group_idx in range(total_groups):
        group_folder = output_path / f"{folder_prefix}{group_idx}"
        count = len(list(group_folder.glob('*')))
        print(f"  {folder_prefix}{group_idx}/  {count} 张图像")
    print("=" * 70)

def process_images(input_dir: str, 
                  output_dir_phase: str = None,
                  output_dir_freq: str = None,
                  enable_phase_classify: bool = True,
                  enable_freq_group: bool = True,
                  copy_mode: bool = True,
                  images_per_group: int = 4):
    """
    主处理函数：根据选择执行一个或两个功能
    
    Parameters:
        input_dir: 输入目录（包含所有图像）
        output_dir_phase: 功能1的输出目录（相位分类）
        output_dir_freq: 功能2的输出目录（频率分组）
        enable_phase_classify: 是否启用功能1（相位分类）
        enable_freq_group: 是否启用功能2（频率分组）
        copy_mode: True=复制文件，False=移动文件
        images_per_group: 每组图像数量（功能2使用）
    """
    print("=" * 70)
    print("🎯 图像分类整理工具")
    print("=" * 70)
    print(f"📁 输入目录: {input_dir}")
    print(f"🔧 文件操作模式: {'复制' if copy_mode else '移动'}")
    print(f"✅ 功能1 - 相位分类: {'启用' if enable_phase_classify else '禁用'}")
    print(f"✅ 功能2 - 频率分组: {'启用' if enable_freq_group else '禁用'}")
    
    if not enable_phase_classify and not enable_freq_group:
        print("\n⚠️  警告：两个功能都未启用，无操作！")
        return
    
    # 执行功能1：相位分类
    if enable_phase_classify:
        if output_dir_phase is None:
            output_dir_phase = str(Path(input_dir).parent / "phase_classified")
        classify_by_phase_pattern(input_dir, output_dir_phase, copy_mode)
    
    # 执行功能2：频率分组
    if enable_freq_group:
        if output_dir_freq is None:
            output_dir_freq = str(Path(input_dir).parent / "freq_grouped")
        group_by_frequency(input_dir, output_dir_freq, copy_mode, images_per_group)
    
    print("\n" + "=" * 70)
    print("🎉 所有任务完成！")
    print("=" * 70)

if __name__ == "__main__":
    # ========================================
    # 配置参数
    # ========================================
    
    # 输入目录（包含所有待分类的图像）
    input_directory = r"F:\test_images"
    
    # 输出目录（可选，不指定则自动生成）
    output_phase_dir = r"F:\output_phase_classified"  # 功能1输出目录
    output_freq_dir = r"F:\output_freq_grouped"       # 功能2输出目录
    
    # 功能开关
    enable_phase_classify = True    # 是否启用功能1：按相位模式分类
    enable_freq_group = True        # 是否启用功能2：按频率组打包
    
    # 其他参数
    copy_files = True               # True=复制文件，False=移动文件
    images_per_group = 4            # 功能2：每组图像数量
    
    # ========================================
    # 执行处理
    # ========================================
    
    print("\n" + "=" * 70)
    print("📋 参数确认")
    print("=" * 70)
    print(f"输入目录: {input_directory}")
    if enable_phase_classify:
        print(f"功能1输出: {output_phase_dir}")
    if enable_freq_group:
        print(f"功能2输出: {output_freq_dir}")
        print(f"每组图像: {images_per_group} 张")
    print(f"操作模式: {'复制文件（保留原文件）' if copy_files else '移动文件（删除原文件）'}")
    print("=" * 70)
    
    # 确认执行
    response = input("\n确认开始处理？(y/n): ").strip().lower()
    if response == 'y' or response == 'yes':
        process_images(
            input_dir=input_directory,
            output_dir_phase=output_phase_dir if enable_phase_classify else None,
            output_dir_freq=output_freq_dir if enable_freq_group else None,
            enable_phase_classify=enable_phase_classify,
            enable_freq_group=enable_freq_group,
            copy_mode=copy_files,
            images_per_group=images_per_group
        )
    else:
        print("❌ 用户取消操作")
        print("💡 提示：如需修改参数，请编辑脚本中的配置部分")


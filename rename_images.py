"""
图像批量重命名工具

功能：
- 将文件夹中的图像按顺序重命名为 0.png, 1.png, 2.png, ...
- 支持预览模式（不实际重命名，只显示将要进行的操作）
- 支持备份原始文件名列表
- 自动识别常见图像格式

使用场景：
- FSA解码需要按顺序命名的图像文件
- 整理混乱命名的图像序列

作者：基于FSA解码需求创建
日期：2025-11-02
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import json
import re


def natural_sort_key(path: Path) -> List:
    """
    自然排序的key函数，使数字按数值大小排序
    
    例如：
    - 字符串排序: 1.png, 10.png, 100.png, 2.png, 20.png
    - 自然排序: 1.png, 2.png, 10.png, 20.png, 100.png
    
    Parameters:
    -----------
    path : Path
        文件路径
        
    Returns:
    --------
    List
        排序用的key列表
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', path.name)]


class ImageRenamer:
    """
    图像批量重命名工具类
    """
    
    # 支持的图像格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    def __init__(self, target_dir: str, output_format: str = "png"):
        """
        初始化重命名器
        
        Parameters:
        -----------
        target_dir : str
            目标文件夹路径
        output_format : str
            输出文件格式，默认为 "png"
        """
        self.target_dir = Path(target_dir)
        self.output_format = output_format.lower().replace('.', '')
        
        if not self.target_dir.exists():
            raise ValueError(f"目标文件夹不存在: {target_dir}")
        
        if not self.target_dir.is_dir():
            raise ValueError(f"路径不是文件夹: {target_dir}")
    
    def get_image_files(self) -> List[Path]:
        """
        获取文件夹中所有的图像文件
        
        Returns:
        --------
        List[Path]
            图像文件路径列表，按自然排序（数字按数值大小排序）
        """
        image_files = []
        
        for file in self.target_dir.iterdir():
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_FORMATS:
                image_files.append(file)
        
        # 使用自然排序（数字按数值大小排序）
        # 例如: 1.png, 2.png, 10.png, 100.png (而不是 1.png, 10.png, 100.png, 2.png)
        image_files.sort(key=natural_sort_key)
        
        return image_files
    
    def preview_rename(self) -> List[Tuple[str, str]]:
        """
        预览重命名操作
        
        Returns:
        --------
        List[Tuple[str, str]]
            (原文件名, 新文件名) 的列表
        """
        image_files = self.get_image_files()
        rename_plan = []
        
        for idx, file in enumerate(image_files):
            old_name = file.name
            new_name = f"{idx}.{self.output_format}"
            rename_plan.append((old_name, new_name))
        
        return rename_plan
    
    def save_backup_list(self, backup_file: str = "rename_backup.json"):
        """
        保存原始文件名列表（用于恢复）
        
        Parameters:
        -----------
        backup_file : str
            备份文件名
        """
        image_files = self.get_image_files()
        backup_path = self.target_dir / backup_file
        
        backup_data = {
            "directory": str(self.target_dir),
            "total_files": len(image_files),
            "file_list": [{"index": idx, "original_name": file.name} 
                         for idx, file in enumerate(image_files)]
        }
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 备份文件已保存: {backup_path}")
    
    def rename_images(self, create_backup: bool = True, 
                     use_temp_rename: bool = True) -> int:
        """
        执行批量重命名
        
        Parameters:
        -----------
        create_backup : bool
            是否创建备份文件
        use_temp_rename : bool
            是否使用临时文件名（避免命名冲突）
            
        Returns:
        --------
        int
            成功重命名的文件数量
        """
        image_files = self.get_image_files()
        
        if len(image_files) == 0:
            print("⚠ 文件夹中没有找到图像文件！")
            return 0
        
        # 创建备份
        if create_backup:
            self.save_backup_list()
        
        success_count = 0
        
        if use_temp_rename:
            # 第一步：重命名为临时文件名（避免冲突）
            print("\n步骤1: 重命名为临时文件名...")
            temp_files = []
            for idx, file in enumerate(image_files):
                temp_name = f"temp_{idx:06d}_{file.suffix}"
                temp_path = self.target_dir / temp_name
                try:
                    file.rename(temp_path)
                    temp_files.append(temp_path)
                    if (idx + 1) % 50 == 0 or idx == len(image_files) - 1:
                        print(f"\r  进度: {idx + 1}/{len(image_files)}", end="")
                except Exception as e:
                    print(f"\n❌ 错误: 无法重命名 {file.name} -> {temp_name}: {e}")
                    # 回滚已重命名的文件
                    for temp_file in temp_files:
                        original_idx = int(temp_file.stem.split('_')[1])
                        temp_file.rename(image_files[original_idx])
                    return success_count
            
            print()  # 换行
            
            # 第二步：重命名为最终文件名
            print("\n步骤2: 重命名为最终文件名...")
            for idx, temp_file in enumerate(temp_files):
                final_name = f"{idx}.{self.output_format}"
                final_path = self.target_dir / final_name
                try:
                    temp_file.rename(final_path)
                    success_count += 1
                    if (idx + 1) % 50 == 0 or idx == len(temp_files) - 1:
                        print(f"\r  进度: {idx + 1}/{len(temp_files)}", end="")
                except Exception as e:
                    print(f"\n❌ 错误: 无法重命名 {temp_file.name} -> {final_name}: {e}")
            
            print()  # 换行
        else:
            # 直接重命名（可能有冲突风险）
            print("\n开始重命名...")
            for idx, file in enumerate(image_files):
                new_name = f"{idx}.{self.output_format}"
                new_path = self.target_dir / new_name
                
                # 检查目标文件是否已存在
                if new_path.exists() and new_path != file:
                    print(f"\n⚠ 警告: 目标文件已存在: {new_name}")
                    print(f"  建议使用 use_temp_rename=True 模式")
                    continue
                
                try:
                    file.rename(new_path)
                    success_count += 1
                    if (idx + 1) % 50 == 0 or idx == len(image_files) - 1:
                        print(f"\r  进度: {idx + 1}/{len(image_files)}", end="")
                except Exception as e:
                    print(f"\n❌ 错误: 无法重命名 {file.name} -> {new_name}: {e}")
            
            print()  # 换行
        
        return success_count


def rename_images_interactive():
    """
    交互式批量重命名工具
    """
    print("="*80)
    print("图像批量重命名工具")
    print("="*80)
    print("功能: 将文件夹中的图像按顺序重命名为 0.png, 1.png, 2.png, ...")
    print("="*80)
    
    # 获取目标文件夹
    print("\n请输入目标文件夹路径:")
    print("示例: E:\\在校学习\\fsa\\data\\fsa_test\\ver")
    target_dir = input("路径: ").strip().strip('"').strip("'")
    
    if not target_dir:
        print("❌ 错误: 路径不能为空")
        return
    
    try:
        renamer = ImageRenamer(target_dir)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return
    
    # 获取图像文件列表
    image_files = renamer.get_image_files()
    
    if len(image_files) == 0:
        print("\n❌ 错误: 文件夹中没有找到图像文件")
        print(f"   支持的格式: {', '.join(ImageRenamer.SUPPORTED_FORMATS)}")
        return
    
    print(f"\n找到 {len(image_files)} 个图像文件")
    print(f"支持的格式: {', '.join(set(f.suffix.lower() for f in image_files))}")
    
    # 选择输出格式
    print("\n输出文件格式:")
    print("1. png (默认)")
    print("2. jpg")
    print("3. 保持原格式")
    format_choice = input("请选择 (1-3，默认1): ").strip()
    
    if format_choice == "2":
        output_format = "jpg"
    elif format_choice == "3":
        output_format = image_files[0].suffix.lower().replace('.', '')
    else:
        output_format = "png"
    
    renamer.output_format = output_format
    
    # 预览重命名
    print("\n" + "="*80)
    print("预览重命名操作")
    print("="*80)
    
    rename_plan = renamer.preview_rename()
    
    # 显示前10个和后10个
    print("\n前10个文件:")
    for old_name, new_name in rename_plan[:10]:
        print(f"  {old_name:40s} → {new_name}")
    
    if len(rename_plan) > 20:
        print(f"\n  ... (省略 {len(rename_plan) - 20} 个文件) ...")
        print(f"\n后10个文件:")
        for old_name, new_name in rename_plan[-10:]:
            print(f"  {old_name:40s} → {new_name}")
    elif len(rename_plan) > 10:
        print()
        for old_name, new_name in rename_plan[10:]:
            print(f"  {old_name:40s} → {new_name}")
    
    # 确认操作
    print("\n" + "="*80)
    print("⚠ 警告: 此操作将重命名所有图像文件！")
    print("   建议先备份重要数据")
    print("="*80)
    
    confirm = input("\n确认执行重命名? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y', '是']:
        print("\n❌ 操作已取消")
        return
    
    # 执行重命名
    print("\n" + "="*80)
    print("开始重命名...")
    print("="*80)
    
    success_count = renamer.rename_images(
        create_backup=True,
        use_temp_rename=True
    )
    
    # 完成
    print("\n" + "="*80)
    print("✓ 重命名完成！")
    print("="*80)
    print(f"成功重命名: {success_count}/{len(image_files)} 个文件")
    print(f"输出格式: .{output_format}")
    print(f"结果保存在: {target_dir}")
    print(f"\n备份文件: {os.path.join(target_dir, 'rename_backup.json')}")
    print("  (如需恢复原始文件名，请保留此文件)")


def rename_images_simple(target_dir: str, output_format: str = "png", 
                        preview_only: bool = False):
    """
    简单的批量重命名函数（供脚本调用）
    
    Parameters:
    -----------
    target_dir : str
        目标文件夹路径
    output_format : str
        输出格式，默认 "png"
    preview_only : bool
        是否仅预览（不实际重命名）
        
    Returns:
    --------
    int
        成功重命名的文件数量（预览模式返回0）
    """
    renamer = ImageRenamer(target_dir, output_format)
    
    if preview_only:
        print("="*80)
        print("预览模式 - 不会实际重命名文件")
        print("="*80)
        rename_plan = renamer.preview_rename()
        print(f"\n将重命名 {len(rename_plan)} 个文件:\n")
        for old_name, new_name in rename_plan:
            print(f"  {old_name:40s} → {new_name}")
        return 0
    else:
        print("="*80)
        print(f"重命名文件夹: {target_dir}")
        print("="*80)
        success_count = renamer.rename_images()
        print(f"\n✓ 完成！成功重命名 {success_count} 个文件")
        return success_count


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        target_dir = sys.argv[1]
        
        # 检查是否为预览模式
        preview = "--preview" in sys.argv or "-p" in sys.argv
        
        # 获取输出格式（跳过 --preview 和 -p 参数）
        output_format = "png"
        for arg in sys.argv[2:]:
            if arg not in ["--preview", "-p"]:
                output_format = arg
                break
        
        try:
            rename_images_simple(target_dir, output_format, preview)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 交互式模式
        rename_images_interactive()


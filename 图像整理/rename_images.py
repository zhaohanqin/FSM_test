"""
图像批量重命名工具 (改进版)

功能：
- 将文件夹中的图像按顺序重命名为 0.png, 1.png, 2.png, ...
- 支持子文件夹批量处理
- 支持预览模式（不实际重命名，只显示将要进行的操作）
- 支持备份原始文件名列表
- 自动识别常见图像格式

使用场景：
- FSA解码需要按顺序命名的图像文件
- 整理混乱命名的图像序列
- 批量处理多个子文件夹的图像

作者：基于FSA解码需求创建
日期：2025-11-02
更新：2025-11-14 - 添加子文件夹支持和配置参数
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json
import re
from concurrent.futures import ThreadPoolExecutor
import threading



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


class BatchImageRenamer:
    """
    批量图像重命名工具类
    用于处理包含多个子文件夹的大文件夹，支持多级文件夹结构
    """
    
    def __init__(self, root_dir: str, output_format: str = "png", max_depth: int = 3):
        """
        初始化批量重命名工具
        
        Parameters:
        -----------
        root_dir : str
            根文件夹路径（包含子文件夹）
        output_format : str
            输出文件格式，默认为 "png"
        max_depth : int
            最大搜索深度，默认3级（1=仅根目录，2=二级目录，3=三级目录）
        """
        self.root_dir = Path(root_dir)
        self.output_format = output_format
        self.max_depth = max_depth
        
        if not self.root_dir.exists():
            raise ValueError(f"根文件夹不存在: {root_dir}")
        
        if not self.root_dir.is_dir():
            raise ValueError(f"路径不是文件夹: {root_dir}")
    
    def find_image_subfolders(self) -> List[Path]:
        """
        查找包含图像的子文件夹（支持多级文件夹结构）
        
        Returns:
        --------
        List[Path]
            包含图像的子文件夹路径列表
        """
        image_folders = []
        
        def search_folders(current_dir: Path, current_depth: int):
            """
            递归搜索包含图像的文件夹
            """
            if current_depth > self.max_depth:
                return
            
            # 检查当前文件夹是否包含图像
            try:
                folder_renamer = ImageRenamer(str(current_dir))
                if folder_renamer.get_image_files():
                    image_folders.append(current_dir)
                    return  # 如果当前文件夹包含图像，不再搜索子文件夹
            except Exception:
                pass
            
            # 如果当前文件夹不包含图像，继续搜索子文件夹
            if current_depth < self.max_depth:
                try:
                    for item in current_dir.iterdir():
                        if item.is_dir():
                            search_folders(item, current_depth + 1)
                except Exception:
                    # 忽略无法访问的文件夹
                    pass
        
        # 从根目录开始搜索
        search_folders(self.root_dir, 1)
        
        return image_folders
    
    def process_folder(self, folder_path: Path, create_backup: bool = True, 
                      use_temp_rename: bool = True) -> Tuple[bool, int]:
        """
        处理单个文件夹
        
        Parameters:
        -----------
        folder_path : Path
            文件夹路径
        create_backup : bool
            是否创建备份
        use_temp_rename : bool
            是否使用临时重命名
            
        Returns:
        --------
        Tuple[bool, int]
            (是否成功, 重命名文件数量)
        """
        try:
            renamer = ImageRenamer(str(folder_path), self.output_format)
            success_count = renamer.rename_images(create_backup, use_temp_rename)
            return True, success_count
        except Exception as e:
            print(f"❌ 处理文件夹 {folder_path} 失败: {e}")
            return False, 0
    
    def batch_rename(self, create_backup: bool = True, 
                    use_temp_rename: bool = True) -> Dict[str, int]:
        """
        批量重命名所有子文件夹的图像
        
        Parameters:
        -----------
        create_backup : bool
            是否创建备份
        use_temp_rename : bool
            是否使用临时重命名
            
        Returns:
        --------
        Dict[str, int]
            各文件夹的处理结果
        """
        image_folders = self.find_image_subfolders()
        
        if not image_folders:
            print("⚠ 没有找到包含图像的文件夹")
            return {}
        
        print(f"找到 {len(image_folders)} 个包含图像的文件夹:")
        for folder in image_folders:
            # 显示相对于根目录的路径
            relative_path = folder.relative_to(self.root_dir) if folder != self.root_dir else Path(".")
            print(f"  - {relative_path} ({folder})")
        
        results = {}
        total_processed = 0
        
        print(f"\n开始批量处理...")
        
        for idx, folder in enumerate(image_folders):
            relative_path = folder.relative_to(self.root_dir) if folder != self.root_dir else Path(".")
            print(f"\n[{idx + 1}/{len(image_folders)}] 处理: {relative_path}")
            print("="*60)
            
            success, count = self.process_folder(folder, create_backup, use_temp_rename)
            results[str(folder)] = count
            total_processed += count
            
            if success:
                print(f"✓ 完成: 重命名 {count} 个文件")
            else:
                print(f"❌ 失败")
        
        print(f"\n" + "="*80)
        print(f"✓ 批量处理完成！")
        print(f"处理文件夹数: {len(image_folders)}")
        print(f"总重命名文件数: {total_processed}")
        print("="*80)
        
        return results


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
    # ==================== 配置参数区域 ====================
    # 直接修改这些参数，然后运行脚本即可
    
    # 目标文件夹路径（支持包含子文件夹的大文件夹）
    TARGET_DIRECTORY = r"E:\images\3-22\v"
    
    # 输出格式
    OUTPUT_FORMAT = "png"  # 可选: png, jpg, 保持原格式
    
    # 是否处理子文件夹
    PROCESS_SUBFOLDERS = True
    
    # 文件夹搜索深度（1=仅根目录，2=二级目录，3=三级目录）
    MAX_SEARCH_DEPTH = 3
    
    # 是否仅预览（不实际重命名）
    PREVIEW_ONLY = False
    
    # 是否创建备份
    CREATE_BACKUP = False
    
    # 是否使用临时重命名（避免冲突）
    USE_TEMP_RENAME = True
    
    # ==================== 配置参数区域结束 ====================
    
    print("="*80)
    print("图像批量重命名工具")
    print("="*80)
    print(f"目标目录: {TARGET_DIRECTORY}")
    print(f"输出格式: {OUTPUT_FORMAT}")
    print(f"处理子文件夹: {PROCESS_SUBFOLDERS}")
    print(f"搜索深度: {MAX_SEARCH_DEPTH}级")
    print(f"预览模式: {PREVIEW_ONLY}")
    print(f"创建备份: {CREATE_BACKUP}")
    print(f"使用临时重命名: {USE_TEMP_RENAME}")
    print("="*80)
    
    if not Path(TARGET_DIRECTORY).exists():
        print(f"❌ 错误: 目标目录不存在: {TARGET_DIRECTORY}")
        exit(1)
    
    try:
        if PROCESS_SUBFOLDERS:
            # 批量处理子文件夹
            batch_renamer = BatchImageRenamer(TARGET_DIRECTORY, OUTPUT_FORMAT, MAX_SEARCH_DEPTH)
            if PREVIEW_ONLY:
                # 预览模式
                folders = batch_renamer.find_image_subfolders()
                print(f"\n将处理 {len(folders)} 个文件夹:")
                for folder in folders:
                    renamer = ImageRenamer(str(folder), OUTPUT_FORMAT)
                    plan = renamer.preview_rename()
                    print(f"\n{folder}:")
                    for old, new in plan[:5]:  # 只显示前5个
                        print(f"  {old} → {new}")
                    if len(plan) > 5:
                        print(f"  ... 还有 {len(plan) - 5} 个文件")
            else:
                batch_renamer.batch_rename(CREATE_BACKUP, USE_TEMP_RENAME)
        else:
            # 单文件夹处理
            if PREVIEW_ONLY:
                rename_images_simple(TARGET_DIRECTORY, OUTPUT_FORMAT, True)
            else:
                rename_images_simple(TARGET_DIRECTORY, OUTPUT_FORMAT, False)
                
        print("\n✓ 处理完成！")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

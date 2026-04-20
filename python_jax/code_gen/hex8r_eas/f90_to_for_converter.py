#!/usr/bin/env python3
"""
Fortran 90 (.f90) to Fortran 77 (.for) 格式转换脚本

转换规则：
1. 注释：! 开头转为 C 开头（固定格式）
2. 缩进：代码必须从第7列开始（前面6个空格）
3. 声明语句：将多变量声明拆分为每行一个变量
4. 续行符：将自由格式的行尾 & 转换为固定格式的第6列 &
"""

import argparse
import re
import sys
from pathlib import Path


def is_comment_line(line):
    """判断是否为注释行"""
    stripped = line.lstrip()
    return stripped.startswith('!')


def is_declaration_line(line):
    """判断是否为变量声明行（包含多个变量的情况）"""
    stripped = line.strip()
    # 匹配 "double precision :: var1, var2, ..."
    # 但排除 intent 声明（函数参数声明）
    if 'intent(' in stripped.lower():
        return False
    return 'double precision' in stripped and '::' in stripped and ',' in stripped


def split_declaration(line):
    """将多变量声明拆分为多行"""
    stripped = line.strip()
    
    # 提取类型和变量列表
    match = re.match(r'(double\s+precision)\s*::\s*(.+)', stripped, re.IGNORECASE)
    if not match:
        return [line]
    
    type_spec = match.group(1)
    vars_str = match.group(2)
    
    # 分割变量名
    var_names = [v.strip() for v in vars_str.split(',')]
    
    # 生成多行声明（每行一个变量，从第7列开始）
    result = []
    for var in var_names:
        result.append(f"      {type_spec} :: {var}")
    
    return result


def find_split_pos(content, max_len):
    """在 content 的前 max_len 个字符中找到一个合适的拆分位置。

    优先在逗号之后拆分，其次在空格处拆分。
    返回拆分位置（content[:pos] 保留在当前行，content[pos:] 移到下一行）。
    """
    # 优先：在最后一个逗号后拆分（保留逗号在当前行）
    last_comma = -1
    for i in range(min(max_len, len(content)) - 1, -1, -1):
        if content[i] == ',':
            last_comma = i
            break

    if last_comma > 10:
        return last_comma + 1

    # 其次：在空格处拆分（避免在标识符中间断开）
    last_space = -1
    for i in range(min(max_len, len(content)) - 1, -1, -1):
        if content[i] == ' ':
            last_space = i
            break

    if last_space > 10:
        return last_space + 1

    # 最后：强制在 max_len 处拆分
    return max_len


def wrap_long_lines(lines, max_col=72):
    """对固定格式 Fortran 的超长行进行自动换行。

    固定格式中：
    - 列 1-5: 标号区（留空）
    - 列 6: 续行标志（空白或 '&'）
    - 列 7-72: 代码区（有效代码最大 66 字符）
    """
    result = []
    for line in lines:
        # 空行、注释行不需要换行
        if not line or line.lstrip().startswith('C') or line.lstrip().startswith('c'):
            result.append(line)
            continue

        if len(line) <= max_col:
            result.append(line)
            continue

        # 提取代码内容（去掉现有的前缀空白）
        content = line.lstrip()

        is_first = True
        while content:
            if is_first:
                prefix = ' ' * 6      # 首行：列 1-6 全空
                avail = max_col - 6   # 代码区 66 字符
                is_first = False
            else:
                prefix = ' ' * 5 + '&' # 续行：列 6 放 &
                avail = max_col - 6   # 代码区 66 字符

            if len(content) <= avail:
                result.append(prefix + content)
                content = ''
            else:
                pos = find_split_pos(content, avail)
                result.append(prefix + content[:pos])
                content = content[pos:]

    return result


def convert_f90_to_for(input_file, output_file):
    """将 .f90 文件转换为 .for 文件"""
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 转换每一行
    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 空行
        if not stripped:
            output_lines.append('')
            i += 1
            continue
        
        # 注释行：! -> C
        if stripped.startswith('!'):
            comment_content = stripped[1:].lstrip()
            output_lines.append(f"C{comment_content}")
            i += 1
            continue
        
        # 声明行：需要检查是否需要拆分
        # 先合并所有续行，再拆分声明
        if is_declaration_line(line):
            full_line = line.rstrip()
            while full_line.endswith('&'):
                full_line = full_line[:-1].rstrip()
                i += 1
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('&'):
                        next_line = next_line[1:].lstrip()
                    full_line += ' ' + next_line
                else:
                    break
            decl_lines = split_declaration(full_line)
            output_lines.extend(decl_lines)
            i += 1
            continue
        
        # 处理普通代码行，包括续行
        # 检查是否以 & 结尾（自由格式的续行）
        if stripped.endswith('&'):
            # 这是一个续行的开始
            # 移除行尾的 &
            code_line = stripped[:-1].rstrip()
            output_lines.append(f"      {code_line}")
            
            # 处理后续的续行
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                
                # 如果下一行也是以 & 结尾
                if next_line.endswith('&'):
                    # 移除行尾的 &，添加续行标记
                    continuation = next_line[:-1].rstrip()
                    output_lines.append(f"     &{continuation}")
                    i += 1
                # 如果下一行不以 & 结尾，这是续行的最后一行
                else:
                    # 添加续行标记
                    output_lines.append(f"     &{next_line}")
                    i += 1
                    break
        else:
            # 普通代码行：从第7列开始（前面6个空格）
            output_lines.append(f"      {stripped}")
            i += 1
    
    # 对超长行进行自动换行（固定格式限制72列）
    output_lines = wrap_long_lines(output_lines)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for out_line in output_lines:
            f.write(out_line + '\n')
    
    print(f"转换完成: {input_file} -> {output_file}")
    print(f"  输入文件行数: {len(lines)}")
    print(f"  输出文件行数: {len(output_lines)}")


def convert_directory(input_dir, output_dir=None, recursive=False):
    """批量转换文件夹中的所有 .f90 文件
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径（可选，默认与输入文件同目录）
        recursive: 是否递归扫描子文件夹
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 输入文件夹不存在: {input_dir}")
        return
    
    if not input_path.is_dir():
        print(f"错误: 输入路径不是文件夹: {input_dir}")
        return
    
    # 获取所有 .f90 文件
    if recursive:
        f90_files = list(input_path.rglob('*.f90'))
    else:
        f90_files = list(input_path.glob('*.f90'))
    
    if not f90_files:
        print(f"警告: 在文件夹 {input_dir} 中未找到 .f90 文件")
        return
    
    print(f"找到 {len(f90_files)} 个 .f90 文件")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    
    for f90_file in f90_files:
        try:
            # 确定输出文件路径
            if output_dir:
                output_path = Path(output_dir)
                if recursive:
                    # 保持相对路径结构
                    rel_path = f90_file.relative_to(input_path)
                    output_file = output_path / rel_path.with_suffix('.for')
                else:
                    output_file = output_path / f90_file.with_suffix('.for').name
                # 确保输出目录存在
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = f90_file.with_suffix('.for')
            
            # 执行转换
            convert_f90_to_for(str(f90_file), str(output_file))
            success_count += 1
        except Exception as e:
            print(f"转换失败: {f90_file}")
            print(f"  错误: {e}")
            fail_count += 1
    
    print("-" * 50)
    print(f"批量转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Fortran 90 (.f90) 到 Fortran 77 (.for) 格式转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
转换规则：
  1. 注释：! 开头转为 C 开头（固定格式）
  2. 缩进：代码必须从第7列开始（前面6个空格）
  3. 声明语句：将多变量声明拆分为每行一个变量
  4. 续行符：将自由格式的行尾 & 转换为固定格式的第6列 &

示例：
  python f90_to_for_converter.py hex8r_op_form_B_gen.f90
  python f90_to_for_converter.py hex8r_op_form_B_gen.f90 output.for
  python f90_to_for_converter.py ./src                     # 批量转换文件夹
  python f90_to_for_converter.py ./src -o ./output         # 批量转换并指定输出目录
  python f90_to_for_converter.py ./src -r                  # 递归扫描子文件夹
        '''
    )
    
    parser.add_argument('input', nargs='?', help='输入的 .f90 文件路径或文件夹路径')
    parser.add_argument('output', nargs='?', help='输出的 .for 文件路径（可选，默认自动生成）')
    parser.add_argument('-o', '--output-dir', help='批量转换时的输出文件夹')
    parser.add_argument('-r', '--recursive', action='store_true', 
                        help='递归扫描子文件夹（仅文件夹模式有效）')
    
    args = parser.parse_args()
    
    # 如果没有提供输入
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    input_path = Path(args.input)
    
    # 检查输入是否存在
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 判断是文件还是文件夹
    if input_path.is_dir():
        # 文件夹模式：批量转换
        output_dir = args.output_dir or args.output
        convert_directory(str(input_path), output_dir, args.recursive)
    else:
        # 文件模式：单个文件转换
        if args.output:
            output_file = args.output
        elif args.output_dir:
            output_file = str(Path(args.output_dir) / input_path.with_suffix('.for').name)
        else:
            output_file = str(input_path.with_suffix('.for'))
        
        convert_f90_to_for(str(input_path), output_file)


if __name__ == '__main__':
    main()

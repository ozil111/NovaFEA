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
        if is_declaration_line(line):
            decl_lines = split_declaration(line)
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
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for out_line in output_lines:
            f.write(out_line + '\n')
    
    print(f"转换完成: {input_file} -> {output_file}")
    print(f"  输入文件行数: {len(lines)}")
    print(f"  输出文件行数: {len(output_lines)}")


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
  python f90_to_for_converter.py *.f90
        '''
    )
    
    parser.add_argument('input', nargs='?', help='输入的 .f90 文件路径')
    parser.add_argument('output', nargs='?', help='输出的 .for 文件路径（可选，默认自动生成）')
    
    args = parser.parse_args()
    
    # 如果没有提供输入文件
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    input_file = args.input
    
    # 如果未指定输出文件，自动生成
    if args.output:
        output_file = args.output
    else:
        # 将 .f90 替换为 .for
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.for'))
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 执行转换
    convert_f90_to_for(input_file, output_file)


if __name__ == '__main__':
    main()

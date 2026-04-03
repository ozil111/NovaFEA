import sympy as sp
from sympy.core.relational import Relational
from sympy.printing.c import C99CodePrinter
from sympy.printing.fortran import FCodePrinter
from sympy.printing.numpy import JaxPrinter
import argparse
import sys
import importlib.util
import importlib
from pathlib import Path

# Add the project root to the Python path to allow finding the 'definitions' module
sys.path.append(str(Path(__file__).parent.resolve()))

from definitions.abc import Element, Material


# ---------------------------------------------------------------------------
# 辅助数据结构：用于 CSE 结果缓存和跨后端共享
# ---------------------------------------------------------------------------
class LoweredChunk:
    """单个 chunk 的 CSE 结果"""
    def __init__(self, chunk_index: int, start_index: int, sub_exprs: list, simplified_outputs: list):
        self.chunk_index = chunk_index
        self.start_index = start_index
        self.sub_exprs = sub_exprs  # list of (Symbol, Expr)
        self.simplified_outputs = simplified_outputs  # list of Expr


class LoweredModel:
    """整个模型经过 CSE lowering 后的结果"""
    def __init__(self, model_name: str, chunk_size: int, chunks: list):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunks = chunks  # list of LoweredChunk


class CachedPrinter:
    """带 memo cache 的 printer 封装器"""
    def __init__(self, printer):
        self.printer = printer
        self.cache = {}

    def doprint(self, expr):
        if expr in self.cache:
            return self.cache[expr]
        result = self.printer.doprint(expr)
        self.cache[expr] = result
        return result


# ---------------------------------------------------------------------------
# 数据容器 + 静态编译分发
# ---------------------------------------------------------------------------
class FEACodePrinter(C99CodePrinter):
    """
    专门为有限元计算优化的代码打印机：
    1. 展开低次幂 pow(x, 2) -> (x*x)
    2. 优化倒数 pow(x, -1) -> (1.0/(x))
    3. 优化平方根和立方根，及分数次幂组合
    """
    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        s_base = self._print(base)
        
        # 处理整数幂 (2, 3, -1, -2)
        if exp.is_Integer:
            if exp == 2:
                return f"({s_base} * {s_base})"
            elif exp == 3:
                return f"({s_base} * {s_base} * {s_base})"
            elif exp == -1:
                return f"(1.0 / ({s_base}))"
            elif exp == -2:
                return f"(1.0 / ({s_base} * {s_base}))"
        
        # 处理分数幂，尽量转化为 sqrt 和 cbrt 的乘除法
        # 注意：这里使用浮点比较处理 SymPy 的 Rational 或 Float
        try:
            val = float(exp)
        except TypeError:
            return super()._print_Pow(expr)

        # 1/2 系列 (sqrt)
        if abs(val - 0.5) < 1e-9:
            return f"sqrt({s_base})"
        if abs(val + 0.5) < 1e-9:
            return f"(1.0 / sqrt({s_base}))"
        
        # 1/3 系列 (cbrt)
        if abs(val - 1.0/3.0) < 1e-7:
            return f"cbrt({s_base})"
        if abs(val + 1.0/3.0) < 1e-7:
            return f"(1.0 / cbrt({s_base}))"
        
        # 2/3 系列
        if abs(val - 2.0/3.0) < 1e-7:
            return f"(cbrt({s_base}) * cbrt({s_base}))"
        if abs(val + 2.0/3.0) < 1e-7:
            return f"(1.0 / (cbrt({s_base}) * cbrt({s_base})))"
            
        # 5/6 系列 (5/6 = 1/2 + 1/3)
        if abs(val - 5.0/6.0) < 1e-7:
            return f"(sqrt({s_base}) * cbrt({s_base}))"
        if abs(val + 5.0/6.0) < 1e-7:
            return f"(1.0 / (sqrt({s_base}) * cbrt({s_base})))"

        # 其他情况回退到标准 pow
        return super()._print_Pow(expr)


class FEAFortranPrinter(FCodePrinter):
    """
    Fortran 90/95+ optimized code printer:
    1. Enforce double precision constants (1.0 -> 1.0d0)
    2. Expand low-order powers to help vectorization
    3. Optimize fractional powers
    """
    def __init__(self, settings=None):
        settings = settings or {}
        settings.update({"standard": 95, "source_format": "free"})
        super().__init__(settings)

    def _print_Piecewise(self, expr):
        # Ensure integer default values in Piecewise are printed as double precision
        # to avoid type mismatch in the Fortran merge() intrinsic.
        if expr.args[-1].cond == True:
            default = expr.args[-1].expr
            if default.is_Integer:
                result = f"{float(default)}d0"
            else:
                result = self._print(default)
            for e, c in reversed(expr.args[:-1]):
                result = "merge(%s, %s, %s)" % (self._print(e), result, self._print(c))
            return result
        return super()._print_Piecewise(expr)

    def _print_Float(self, expr):
        # Keep all floating constants in double precision.
        res = super()._print_Float(expr)
        return res + "d0" if "d" not in res.lower() and "e" not in res.lower() else res

    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        s_base = self._print(base)
        s_exp = self._print(exp)

        # Integer powers
        if exp.is_Integer:
            if exp == 2:
                return f"({s_base} * {s_base})"
            if exp == 3:
                return f"({s_base} * {s_base} * {s_base})"
            if exp == -1:
                return f"(1.0d0 / ({s_base}))"
            if exp == -2:
                return f"(1.0d0 / ({s_base} * {s_base}))"

        try:
            val = float(exp)
        except TypeError:
            return super()._print_Pow(expr)

        # Fractional powers
        if abs(val - 0.5) < 1e-9:
            return f"sqrt({s_base})"
        if abs(val + 0.5) < 1e-9:
            return f"(1.0d0 / sqrt({s_base}))"
        if abs(val - 1.0 / 3.0) < 1e-7:
            return f"({s_base}**(1.0d0/3.0d0))"
        if abs(val + 1.0 / 3.0) < 1e-7:
            return f"(1.0d0 / ({s_base}**(1.0d0/3.0d0)))"
        if abs(val - 2.0 / 3.0) < 1e-7:
            return f"({s_base}**(2.0d0/3.0d0))"
        if abs(val + 2.0 / 3.0) < 1e-7:
            return f"(1.0d0 / ({s_base}**(2.0d0/3.0d0)))"
        if abs(val - 5.0 / 6.0) < 1e-7:
            return f"({s_base}**(5.0d0/6.0d0))"
        if abs(val + 5.0 / 6.0) < 1e-7:
            return f"(1.0d0 / ({s_base}**(5.0d0/6.0d0)))"

        # Parenthesize the exponent to preserve precedence in Fortran.
        return f"({s_base}**({s_exp}))"


class MathModel:
    """数据容器：存储数学定义"""
    def __init__(self, inputs, outputs, name="kernel", input_names=None, output_names=None, is_operator=False):
        self.inputs = inputs   # SymPy 符号列表
        self.outputs = outputs # SymPy 表达式列表
        self.name = name
        self.input_names = input_names or [str(s) for s in inputs]
        self.output_names = output_names or [f"out[{i}]" for i in range(len(outputs))]
        self.is_operator = is_operator # 是否作为算子生成（可能包含SIMD优化等）


class FEACompiler:
    # =========================================================================
    # 公共 Lower 阶段：将 MathModel 转换为 LoweredModel，执行 CSE
    # =========================================================================
    @staticmethod
    def lower_model(model: MathModel, chunk_size: int) -> LoweredModel:
        """执行 CSE lowering，返回可被多个后端共享的 LoweredModel"""
        outputs = model.outputs
        chunks = []
        
        for start in range(0, len(outputs), chunk_size):
            chunk_index = start // chunk_size
            chunk = outputs[start:start + chunk_size]
            sub_exprs, simplified_chunk = sp.cse(
                chunk,
                symbols=sp.numbered_symbols(f"v_{chunk_index}_")
            )
            chunks.append(
                LoweredChunk(
                    chunk_index=chunk_index,
                    start_index=start,
                    sub_exprs=sub_exprs,
                    simplified_outputs=simplified_chunk
                )
            )
        
        return LoweredModel(model.name, chunk_size, chunks)
    
    # =========================================================================
    # Chunk Size 策略：根据模型规模和目标平台决定 chunk size
    # =========================================================================
    @staticmethod
    def resolve_chunk_size(model: MathModel, target: str, user_chunk_size=None, strategy="auto") -> int:
        """
        决定 CSE chunk size 的策略。
        
        Args:
            model: 数学模型
            target: 目标平台 (jax/cpp/cuda/fortran等)
            user_chunk_size: 用户通过 CLI 指定的 chunk size
            strategy: 策略模式 ("auto" 或 "fixed")
        
        Returns:
            最终的 chunk size
        """
        if user_chunk_size is not None:
            return user_chunk_size
        
        nout = len(model.outputs)
        target = target.lower()
        
        # fixed 模式：使用各后端的固定默认值
        if strategy == "fixed":
            if target == "jax":
                return 50
            if target in ("cpp", "c++", "cuda", "fortran"):
                return 24
            return 24
        
        # auto 模式：根据输出规模自动调整
        if strategy == "auto":
            if target == "jax":
                if nout <= 64:
                    return 64
                elif nout <= 256:
                    return 48
                else:
                    return 32
            
            # cpp/cuda/fortran 的自适应策略
            if nout <= 32:
                return 32
            elif nout <= 128:
                return 24
            elif nout <= 512:
                return 16
            else:
                return 8
        
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # =========================================================================
    # C++/CUDA 兼容性宏：跨平台支持 GCC/Clang/MSVC/CUDA
    # =========================================================================
    @staticmethod
    def _cpp_cuda_compat_macros() -> str:
        """返回统一的 C++/CUDA 跨平台兼容性宏定义"""
        return r"""
#if defined(__CUDACC__)
  #define FEA_DEVICE __device__
  #define FEA_HOST __host__
  #define FEA_HOST_DEVICE __host__ __device__
  #define FEA_RESTRICT __restrict__
#else
  #define FEA_DEVICE
  #define FEA_HOST
  #define FEA_HOST_DEVICE
  #if defined(_WIN32) || defined(_WIN64)
    #if defined(_MSC_VER)
      #define FEA_RESTRICT __restrict
    #else
      #define FEA_RESTRICT __restrict__
    #endif
  #else
    #if defined(__GNUC__) || defined(__clang__)
      #define FEA_RESTRICT __restrict__
    #else
      #define FEA_RESTRICT
    #endif
  #endif
#endif

#if defined(_MSC_VER)
  #define FEA_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define FEA_ALWAYS_INLINE inline __attribute__((always_inline))
#else
  #define FEA_ALWAYS_INLINE inline
#endif
"""
    
    # =========================================================================
    # 核心编译接口
    # =========================================================================
    @staticmethod
    def compile(model: MathModel, target: str, chunk_size=None, cse_strategy="auto", lowered=None):
        """
        核心分发器：输入 MathModel + target，输出 cpp/cuda/jax/fortran 源码字符串。
        
        Args:
            model: 数学模型
            target: 目标平台 ('jax', 'cpp', 'cuda', 'fortran')
            chunk_size: 用户指定的 chunk size (可选)
            cse_strategy: CSE 策略 ('auto' 或 'fixed')
            lowered: 预先 lowered 的结果 (可选，用于多后端共享)
        """
        target = target.lower()
        if target == 'jax':
            return FEACompiler._to_jax(model, lowered=lowered, chunk_size=chunk_size, cse_strategy=cse_strategy)
        elif target in ['cpp', 'c++']:
            return FEACompiler._to_source(model, is_cuda=False, lowered=lowered, 
                                         chunk_size=chunk_size, cse_strategy=cse_strategy)
        elif target == 'cuda':
            return FEACompiler._to_source(model, is_cuda=True, lowered=lowered,
                                         chunk_size=chunk_size, cse_strategy=cse_strategy)
        elif target == 'fortran':
            return FEACompiler._to_fortran(model, lowered=lowered,
                                          chunk_size=chunk_size, cse_strategy=cse_strategy)
        else:
            raise ValueError(f"Unknown target: {target}")

    @staticmethod
    def compile_all(model: MathModel, chunk_size=None, cse_strategy="auto"):
        """
        一次性生成 jax/cpp/cuda/fortran 四种目标源码。
        
        统一管理 lower 行为：
        - 如果所有 target 使用相同的 chunk size，共享一份 lowered
        - 如果 JAX 和 cpp/cuda/fortran 使用不同的 chunk size，分别生成 jax_lowered 和 shared_lowered
        
        Args:
            model: 数学模型
            chunk_size: 用户指定的 chunk size (可选)
            cse_strategy: CSE 策略 ('auto' 或 'fixed')
        
        Returns:
            dict: {'jax': code, 'cpp': code, 'cuda': code, 'fortran': code}
        """
        # 决定各 target 的 chunk size
        cpp_chunk = FEACompiler.resolve_chunk_size(model, "cpp", chunk_size, cse_strategy)
        jax_chunk = FEACompiler.resolve_chunk_size(model, "jax", chunk_size, cse_strategy)
        
        # 生成 shared lowered 给 cpp/cuda/fortran
        shared_lowered = FEACompiler.lower_model(model, cpp_chunk)
        
        # 决定 JAX 是否共享 lowered
        if jax_chunk == cpp_chunk:
            jax_lowered = shared_lowered
        else:
            jax_lowered = FEACompiler.lower_model(model, jax_chunk)
        
        return {
            "jax": FEACompiler._to_jax(model, lowered=jax_lowered, chunk_size=jax_chunk, cse_strategy=cse_strategy),
            "cpp": FEACompiler._to_source(model, is_cuda=False, lowered=shared_lowered, chunk_size=cpp_chunk, cse_strategy=cse_strategy),
            "cuda": FEACompiler._to_source(model, is_cuda=True, lowered=shared_lowered, chunk_size=cpp_chunk, cse_strategy=cse_strategy),
            "fortran": FEACompiler._to_fortran(model, lowered=shared_lowered, chunk_size=cpp_chunk, cse_strategy=cse_strategy),
        }

    @staticmethod
    def _to_jax(model, lowered=None, chunk_size=None, cse_strategy="auto"):
        """
        生成 JAX 源码（.py），采用分块 CSE 优化。
        
        Args:
            model: 数学模型
            lowered: 预先 lowered 的 LoweredModel (可选，用于多后端共享)
            chunk_size: 用户指定的 chunk size (可选)
            cse_strategy: CSE 策略 ('auto' 或 'fixed')
        """
        # 如果没有提供 lowered 结果，则自行 lower
        if lowered is None:
            chunk_size = FEACompiler.resolve_chunk_size(model, "jax", chunk_size, cse_strategy)
            lowered = FEACompiler.lower_model(model, chunk_size)
        
        lines = [
            '"""Generated by sympy_codegen.py. Do not edit."""',
            "import jax.numpy as jnp",
            "",
            "",
            f"def compute_{model.name}(in_flat):",
            f'    """',
            f'    Compute the {model.name} kernel.',
            f'    ',
            f'    Args:',
            f'        in_flat: Flattened input array, size {len(model.inputs)}',
            f'    ',
            f'    Returns:',
            f'        Flattened output array, size {len(model.outputs)}',
            f'    ',
            f'    Input layout:',
        ]
        
        # 添加输入信息
        for i, name in enumerate(model.input_names):
            lines.append(f"    '       - in_flat[{i}]: {name}")
        
        lines.append(f"    '")
        lines.append(f"    '    Output layout:")
        
        # 添加输出信息
        for i, name in enumerate(model.output_names):
            lines.append(f"    '       - out[{i}]: {name}")
        
        lines.append(f'    """')
        
        # Unpack inputs IF they are valid identifiers (e.g. xi, c0)
        # If they are like "in[0]", we'll handle them via string replacement later
        for i, sym in enumerate(model.inputs):
            s = str(sym)
            is_ident = s.isidentifier()
            # print(f"DEBUG: sym={s}, is_ident={is_ident}")
            if is_ident:
                lines.append(f"    {s} = in_flat[{i}]")
        
        lines.append("")
        
        printer = CachedPrinter(JaxPrinter())
        all_simplified_outputs = []
        
        # 使用 lowered 结果
        for chunk in lowered.chunks:
            for var, expr in chunk.sub_exprs:
                lines.append(f"    {var} = {printer.doprint(expr)}")
            
            all_simplified_outputs.extend(chunk.simplified_outputs)

        lines.append("")
        lines.append("    # --- Output ---")
        out_parts = [printer.doprint(e) for e in all_simplified_outputs]
        lines.append(f"    return ({','.join(out_parts)})")
        
        src = "\n".join(lines)
        # Final cleanup for JAX and handle C-style inputs
        src = src.replace("jax.numpy.", "jnp.").replace("in[", "in_flat[")
        return src

    @staticmethod
    def _to_source(model, is_cuda=False, lowered=None, chunk_size=None, cse_strategy="auto"):
        """
        生成 C++/CUDA 源码，采用分块 CSE 优化及算子化增强。
        
        Args:
            model: 数学模型
            is_cuda: 是否为 CUDA 目标
            lowered: 预先 lowered 的 LoweredModel (可选，用于多后端共享)
            chunk_size: 用户指定的 chunk size (可选)
            cse_strategy: CSE 策略 ('auto' 或 'fixed')
        """
        # 如果没有提供 lowered 结果，则自行 lower
        if lowered is None:
            chunk_size = FEACompiler.resolve_chunk_size(model, "cuda" if is_cuda else "cpp", 
                                                       chunk_size, cse_strategy)
            lowered = FEACompiler.lower_model(model, chunk_size)
        
        # --- Generate Comments ---
        comment_lines = ["/**"]
        comment_lines.append(f" * @brief Computes the {model.name} kernel.")
        if model.is_operator:
            comment_lines.append(" * @note This is an optimized operator kernel.")
        comment_lines.append(" * ")
        comment_lines.append(" * @param in Input array (const double*). Layout:")
        
        for i, name in enumerate(model.input_names):
            comment_lines.append(f" *   - in[{i}]: {name}")

        comment_lines.append(" * ")
        comment_lines.append(" * @param out Output array (double*). Layout:")
        
        # 列出每个输出的详细信息
        for i, name in enumerate(model.output_names):
            comment_lines.append(f" *   - out[{i}]: {name}")
        
        comment_lines.append(" */")
        comment_block = "\n".join(comment_lines)

        # --- Generate Function Body ---
        body_lines = []

        # 解包输入变量
        for i, sym in enumerate(model.inputs):
            s = str(sym)
            # 检查是否是合法标识符（如 coord_2_3），如果是则解包
            if s.isidentifier():
                body_lines.append(f"    double {s} = in[{i}];")

        body_lines.append("")

        # 初始化带缓存的 Printer
        printer = CachedPrinter(FEACodePrinter())

        # 使用 lowered 结果
        for chunk in lowered.chunks:
            body_lines.append(f"\n    // --- Chunk {chunk.chunk_index} ---")
            
            for var, expr in chunk.sub_exprs:
                body_lines.append(f"    double {var} = {printer.doprint(expr)};")
            
            for j, out_expr in enumerate(chunk.simplified_outputs):
                body_lines.append(f"    out[{chunk.start_index + j}] = {printer.doprint(out_expr)};")

        body = "\n".join(body_lines)
        
        # 统一使用兼容宏体系
        prefix = FEACompiler._cpp_cuda_compat_macros() + "\n"
        
        if is_cuda:
            # CUDA 使用 FEA_DEVICE 宏
            func_type = "FEA_DEVICE FEA_ALWAYS_INLINE void"
            signature = f"{func_type} compute_{model.name}(const double* FEA_RESTRICT in, double* FEA_RESTRICT out)"
        else:
            # C++ 使用 FEA_ALWAYS_INLINE 宏
            func_type = "FEA_ALWAYS_INLINE void"
            signature = f"{func_type} compute_{model.name}(const double* FEA_RESTRICT in, double* FEA_RESTRICT out)"
        
        return f"{prefix}{comment_block}\n{signature} {{ \n{body}\n}}"



    @staticmethod
    def _to_fortran(model, lowered=None, chunk_size=None, cse_strategy="auto"):
        """生成 Fortran 源码,支持分块 CSE 优化。声明和赋值必须分离。"""
        # 如果没有提供 lowered 结果，则自行 lower
        if lowered is None:
            chunk_size = FEACompiler.resolve_chunk_size(model, "fortran", chunk_size, cse_strategy)
            lowered = FEACompiler.lower_model(model, chunk_size)
        
        printer = CachedPrinter(FEAFortranPrinter())

        lines = [
            "! Generated by sympy_codegen.py. Do not edit.",
            "!",
            f"! Subroutine: compute_{model.name}",
            "!",
            "! Input array layout (in_vec):",
        ]
        
        # 添加输入信息
        for i, name in enumerate(model.input_names):
            lines.append(f"!   in_vec({i + 1}): {name}")
        
        lines.append("!")
        lines.append("! Output array layout (out_vec):")
        
        # 添加输出信息
        for i, name in enumerate(model.output_names):
            lines.append(f"!   out_vec({i + 1}): {name}")
        
        lines.extend([
            "!",
            f"subroutine compute_{model.name}(in_vec, out_vec)",
            "    implicit none",
            f"    double precision, intent(in)  :: in_vec({len(model.inputs)})",
            f"    double precision, intent(out) :: out_vec({len(model.outputs)})",
            "    ! --- Unpack inputs ---",
        ])

        # Unpack input array to named variables
        input_vars = []
        for i, sym in enumerate(model.inputs):
            s = str(sym)
            if s.isidentifier():
                input_vars.append(s)

        # First, declare all input variables
        if input_vars:
            lines.append(f"    double precision :: {', '.join(input_vars)}")

        # Then assign values
        for i, sym in enumerate(model.inputs):
            s = str(sym)
            if s.isidentifier():
                lines.append(f"    {s} = in_vec({i + 1})")

        lines.append("    ! --- Local Variables for CSE ---")

        # 使用 lowered 结果
        for chunk in lowered.chunks:
            lines.append(f"    ! Chunk {chunk.chunk_index}")
            if chunk.sub_exprs:
                lines.append("    block")

                # Separate variables by type: logical for comparisons, double precision otherwise
                dp_vars = []
                log_vars = []
                for var, expr in chunk.sub_exprs:
                    if isinstance(expr, Relational):
                        log_vars.append(str(var))
                    else:
                        dp_vars.append(str(var))

                if dp_vars:
                    lines.append(f"        double precision :: {', '.join(dp_vars)}")
                if log_vars:
                    lines.append(f"        logical :: {', '.join(log_vars)}")

                # Then assign values
                for var, expr in chunk.sub_exprs:
                    lines.append(f"        {var} = {printer.doprint(expr)}")

            for j, out_expr in enumerate(chunk.simplified_outputs):
                # Fortran arrays are 1-based.
                lines.append(f"        out_vec({chunk.start_index + j + 1}) = {printer.doprint(out_expr)}")

            if chunk.sub_exprs:
                lines.append("    end block")

        lines.append(f"end subroutine compute_{model.name}")
        return "\n".join(lines)

    @staticmethod
    def compile_element(element: Element, target: str, chunk_size=None, cse_strategy="auto"):
        """
        Special compiler for Elements: supports both single-kernel and operator-based generation.
        """
        operators = element.get_stiffness_operators()
        if operators:
            # Generate multiple operator kernels
            generated = {}
            for op_model in operators:
                generated[op_model.name] = FEACompiler.compile(op_model, target, 
                                                               chunk_size=chunk_size, cse_strategy=cse_strategy)
            return generated
        else:
            # Traditional single kernel
            model = element.get_stiffness_model()
            return {model.name: FEACompiler.compile(model, target, 
                                                   chunk_size=chunk_size, cse_strategy=cse_strategy)}


# ---------------------------------------------------------------------------
# 动态模型加载
# ---------------------------------------------------------------------------
def _load_class(module_path: str, class_name: str):
    """动态加载类"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find class '{class_name}' in module '{module_path}'.\nError: {e}")

def load_element(name: str) -> Element:
    """Loads an element class from the definitions.elements directory."""
    class_name = name.capitalize()
    module_path = f"definitions.elements.{name}"
    return _load_class(module_path, class_name)()

def load_material(name: str) -> Material:
    """Loads a material class from the definitions.materials directory."""
    class_name = name.capitalize()
    module_path = f"definitions.materials.{name}"
    return _load_class(module_path, class_name)()


def _default_output(model_name: str, target: str) -> str:
    t = target.lower()
    if t == "jax":
        return f"{model_name}_gen.py"
    if t in ("cpp", "c++"):
        return f"{model_name}_gen.cpp"
    if t == "cuda":
        return f"{model_name}_gen.cu"
    if t == "fortran":
        return f"{model_name}_gen.f90"
    return f"{model_name}_{t}.txt"

def main():
    parser = argparse.ArgumentParser(
        description="SymPy FEA 代码生成器 (混合解耦架构)"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["constitutive", "stiffness", "mass", "custom"],
        help="生成任务: 'constitutive' (材料D矩阵), 'stiffness' (单元Ke矩阵), 'mass' (质量矩阵), 或 'custom' (自定义数学模型)",
    )
    parser.add_argument(
        "--element", "-e",
        help="单元名称 (e.g., 'tet4'), required for --task=stiffness",
    )
    parser.add_argument(
        "--material", "-m",
        help="材料名称 (e.g., 'isotropic'), required for --task=constitutive",
    )
    parser.add_argument(
        "--script", "-s",
        help="Python 脚本路径 (用于 --task=custom). 脚本中需要提供 get_model() 函数返回 MathModel.",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        choices=["jax", "cpp", "cuda", "fortran", "all"],
        help="目标语言：jax / cpp / cuda / fortran / all",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出文件路径（默认根据任务和名称生成）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="CSE chunk size. 如果省略，则使用 cse-strategy 决定。",
    )
    parser.add_argument(
        "--cse-strategy",
        choices=["auto", "fixed"],
        default="auto",
        help="CSE chunk sizing 策略。'auto' 根据输出规模自动调整，'fixed' 使用固定默认值。",
    )
    args = parser.parse_args()

    if args.task == "constitutive":
        if not args.material:
            parser.error("--material is required for --task=constitutive")
        material = load_material(args.material)
        model = material.get_constitutive_model()
        models_to_compile = {model.name: model}
        
    elif args.task == "stiffness":
        if not args.element:
            parser.error("--element is required for --task=stiffness")
        element = load_element(args.element)
        operators = element.get_stiffness_operators()
        if operators:
            models_to_compile = {op.name: op for op in operators}
        else:
            m = element.get_stiffness_model()
            models_to_compile = {m.name: m}

    elif args.task == "mass":
        if not args.element:
            parser.error("--element is required for --task=mass")
        element = load_element(args.element)
        operators = element.get_mass_operators()
        if operators:
            models_to_compile = {op.name: op for op in operators}
        else:
            # Fallback if no specific mass model is defined, though mass usually has operators
            parser.error(f"No mass operators defined for element: {args.element}")
            
    elif args.task == "custom":
        if not args.script:
            parser.error("--script is required for --task=custom")
        script_path = Path(args.script)
        if not script_path.exists():
            parser.error(f"Script file not found: {script_path}")
            
        # Dynamically load the script
        spec = importlib.util.spec_from_file_location("custom_script", str(script_path))
        custom_mod = importlib.util.module_from_spec(spec)
        sys.modules["custom_script"] = custom_mod
        spec.loader.exec_module(custom_mod)
        
        if not hasattr(custom_mod, "get_model"):
            parser.error(f"Script {script_path} must define a 'get_model()' function.")
            
        models = custom_mod.get_model()
        if type(models).__name__ == "MathModel":
            models_to_compile = {models.name: models}
        elif isinstance(models, list) and all(type(m).__name__ == "MathModel" for m in models):
            models_to_compile = {m.name: m for m in models}
        else:
            parser.error(f"get_model() must return a MathModel or a list of MathModels. Got: {type(models)}")

    # ---------------- Compile Models ----------------
    for name, model in models_to_compile.items():
        if args.target == "all":
            # --target all: 使用 compile_all 实现真正的共享 CSE
            generated = FEACompiler.compile_all(
                model,
                chunk_size=args.chunk_size,
                cse_strategy=args.cse_strategy,
            )
            for t, code in generated.items():
                out_path = Path(args.output or ".") / _default_output(name, t)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(f"Generated: {out_path}")
        else:
            # 单一目标编译
            code = FEACompiler.compile(
                model,
                args.target,
                chunk_size=args.chunk_size,
                cse_strategy=args.cse_strategy,
            )
            out_path = Path(args.output or ".") / _default_output(name, args.target)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()

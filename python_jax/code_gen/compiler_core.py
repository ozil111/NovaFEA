from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import sympy as sp
from jinja2 import Environment, FileSystemLoader


@dataclass
class MathModel:
    name: str
    inputs: List[sp.Symbol]
    outputs: List[sp.Expr]


@dataclass
class Instruction:
    op: str
    dest: str
    srcs: List[str] = field(default_factory=list)
    const_val: Optional[float] = None
    dead_vars: List[str] = field(default_factory=list)


class IRGenerator:
    def __init__(self) -> None:
        self.instructions: List[Instruction] = []
        self._tmp_idx = 0
        self._const_cache: Dict[sp.Expr, str] = {}
        self._input_map: Dict[sp.Symbol, str] = {}

    def _new_tmp(self) -> str:
        name = f"v_{self._tmp_idx}"
        self._tmp_idx += 1
        return name

    def _const_name(self, val: sp.Expr) -> str:
        if val not in self._const_cache:
            text = str(float(val)).replace("-", "m").replace(".", "_")
            self._const_cache[val] = f"c_{text}"
        return self._const_cache[val]

    def _emit_const(self, val: sp.Expr) -> str:
        cname = self._const_name(val)
        if not any(inst.dest == cname for inst in self.instructions):
            self.instructions.append(
                Instruction(op="CONST", dest=cname, const_val=float(val))
            )
        return cname

    def _emit_input(self, sym: sp.Symbol) -> str:
        tmp = self._new_tmp()
        self.instructions.append(Instruction(op="LOAD", dest=tmp, srcs=[str(sym)]))
        return tmp

    def _lower_expr(self, expr: sp.Expr) -> str:
        if expr.is_Number:
            return self._emit_const(expr)
        if expr.is_Symbol:
            return self._emit_input(expr)

        if isinstance(expr, sp.Add):
            src = self._lower_expr(expr.args[0])
            for arg in expr.args[1:]:
                rhs = self._lower_expr(arg)
                dst = self._new_tmp()
                self.instructions.append(Instruction(op="ADD", dest=dst, srcs=[src, rhs]))
                src = dst
            return src

        if isinstance(expr, sp.Mul):
            src = self._lower_expr(expr.args[0])
            for arg in expr.args[1:]:
                rhs = self._lower_expr(arg)
                dst = self._new_tmp()
                self.instructions.append(Instruction(op="MUL", dest=dst, srcs=[src, rhs]))
                src = dst
            return src

        if isinstance(expr, sp.Pow):
            base = self._lower_expr(expr.args[0])
            exp = self._lower_expr(expr.args[1])
            dst = self._new_tmp()
            self.instructions.append(Instruction(op="POW", dest=dst, srcs=[base, exp]))
            return dst

        if len(expr.args) == 1:
            arg0 = self._lower_expr(expr.args[0])
            dst = self._new_tmp()
            self.instructions.append(
                Instruction(op=type(expr).__name__.upper(), dest=dst, srcs=[arg0])
            )
            return dst

        raise TypeError(f"Unsupported SymPy expression type: {type(expr)}")

    def _mark_dead_vars(self) -> None:
        last_use: Dict[str, int] = {}
        for idx, inst in enumerate(self.instructions):
            for src in inst.srcs:
                last_use[src] = idx

        for idx, inst in enumerate(self.instructions):
            dead: List[str] = []
            for var, use_idx in last_use.items():
                if use_idx == idx and (var.startswith("v_") or var.startswith("c_")):
                    dead.append(var)
            inst.dead_vars = sorted(dead)

    def generate_ir(self, model: MathModel) -> List[Instruction]:
        self.instructions = []
        self._tmp_idx = 0
        self._const_cache = {}
        self._input_map = {}

        for out_idx, out_expr in enumerate(model.outputs):
            out_var = self._lower_expr(out_expr)
            self.instructions.append(
                Instruction(op="STORE", dest=f"out_{out_idx}", srcs=[out_var])
            )

        self._mark_dead_vars()
        return self.instructions

    @staticmethod
    def format_ir(instructions: List[Instruction]) -> str:
        lines: List[str] = []
        for idx, inst in enumerate(instructions):
            parts = [f"{idx:04d}", inst.op, inst.dest]
            if inst.srcs:
                parts.append(f"srcs={inst.srcs}")
            if inst.const_val is not None:
                parts.append(f"const={inst.const_val}")
            if inst.dead_vars:
                parts.append(f"dead={inst.dead_vars}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)


class RegisterAllocator:
    def __init__(self, physical_regs: Optional[List[str]] = None) -> None:
        if physical_regs is None:
            # Keep xmm14/xmm15 as reserved.
            physical_regs = [f"xmm{i}" for i in range(14)]
        self._all_regs = list(physical_regs)
        self._free_regs: List[str] = []
        self._vreg_to_preg: Dict[str, str] = {}

    def _is_allocatable_vreg(self, name: str) -> bool:
        return name.startswith("v_")

    def _alloc(self, vreg: str) -> str:
        if vreg in self._vreg_to_preg:
            return self._vreg_to_preg[vreg]
        if not self._free_regs:
            raise NotImplementedError("Register Spill needed")
        preg = self._free_regs.pop()
        self._vreg_to_preg[vreg] = preg
        return preg

    def _map_read(self, name: str) -> str:
        if self._is_allocatable_vreg(name):
            if name not in self._vreg_to_preg:
                raise ValueError(f"Use-before-define for virtual register: {name}")
            return self._vreg_to_preg[name]
        return name

    def _free_dead(self, dead_vars: List[str]) -> List[str]:
        dead_regs: List[str] = []
        for vreg in dead_vars:
            preg = self._vreg_to_preg.pop(vreg, None)
            if preg is not None:
                self._free_regs.append(preg)
                dead_regs.append(preg)
        return dead_regs

    def allocate(self, instructions: List[Instruction]) -> List[Instruction]:
        self._free_regs = list(self._all_regs)
        self._vreg_to_preg = {}

        allocated: List[Instruction] = []
        for inst in instructions:
            mapped_srcs = [self._map_read(src) for src in inst.srcs]
            # After reading srcs, dead vars can be released immediately.
            dead_regs = self._free_dead(inst.dead_vars)
            mapped_dest = inst.dest

            if self._is_allocatable_vreg(inst.dest):
                mapped_dest = self._alloc(inst.dest)

            allocated.append(
                Instruction(
                    op=inst.op,
                    dest=mapped_dest,
                    srcs=mapped_srcs,
                    const_val=inst.const_val,
                    dead_vars=sorted(dead_regs),
                )
            )

        return allocated


class PeachPyBackend:
    def __init__(self, template_path: Optional[Path] = None) -> None:
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "peachpy_kernel.py.j2"
        self.template_path = Path(template_path)
        self._const_table: Dict[str, float] = {}
        self._live_regs: set[str] = set()

    @staticmethod
    def _is_reg(name: str) -> bool:
        return name.startswith("xmm")

    @staticmethod
    def _input_offset(name: str, input_name_to_idx: Dict[str, int]) -> int:
        if name not in input_name_to_idx:
            raise ValueError(f"Unknown input symbol in LOAD: {name}")
        return input_name_to_idx[name] * 8

    def _value_ref(self, name: str, input_name_to_idx: Dict[str, int]) -> str:
        if self._is_reg(name):
            return name
        if name.startswith("c_"):
            if name not in self._const_table:
                raise ValueError(f"Constant {name} used before CONST definition")
            return f"Constant.float64({self._const_table[name]})"
        if name in input_name_to_idx:
            return f"[reg_in + {self._input_offset(name, input_name_to_idx)}]"
        raise ValueError(f"Unsupported source value: {name}")

    def _emit_binary(
        self,
        op: str,
        dst: str,
        srcs: List[str],
        input_name_to_idx: Dict[str, int],
    ) -> List[str]:
        if len(srcs) != 2:
            raise ValueError(f"{op} expects 2 srcs, got {len(srcs)}")
        a = self._value_ref(srcs[0], input_name_to_idx)
        b = self._value_ref(srcs[1], input_name_to_idx)

        lines: List[str] = []
        if op == "ADD":
            if srcs[0] == dst:
                lines.append(f"ADDSD({dst}, {b})")
            elif srcs[1] == dst:
                lines.append(f"ADDSD({dst}, {a})")
            else:
                lines.append(f"MOVSD({dst}, {a})")
                lines.append(f"ADDSD({dst}, {b})")
        elif op == "MUL":
            if srcs[0] == dst:
                lines.append(f"MULSD({dst}, {b})")
            elif srcs[1] == dst:
                lines.append(f"MULSD({dst}, {a})")
            else:
                lines.append(f"MOVSD({dst}, {a})")
                lines.append(f"MULSD({dst}, {b})")
        elif op == "SUB":
            if a != dst:
                lines.append(f"MOVSD({dst}, {a})")
            lines.append(f"SUBSD({dst}, {b})")
        elif op == "DIV":
            if a != dst:
                lines.append(f"MOVSD({dst}, {a})")
            lines.append(f"DIVSD({dst}, {b})")
        elif op == "POW":
            if a != dst:
                lines.append(f"MOVSD({dst}, {a})")
            if srcs[1].startswith("c_"):
                const_val = self._const_table.get(srcs[1])
                if const_val == 2.0:
                    lines.append(f"MULSD({dst}, {dst})")
                elif const_val == 0.5:
                    lines.append(f"SQRTSD({dst}, {dst})")
                elif const_val == -1.0:
                    lines = [f"MOVSD({dst}, Constant.float64(1.0))", f"DIVSD({dst}, {a})"]
                else:
                    lines.append(f"# Unsupported POW exponent {const_val} for {dst}")
            else:
                lines.append(f"# Unsupported dynamic POW exponent for {dst}")
        else:
            raise ValueError(f"Unsupported binary op: {op}")
        return lines

    def _translate_instruction(
        self, inst: Instruction, input_name_to_idx: Dict[str, int]
    ) -> List[str]:
        op = inst.op
        if op == "CONST":
            if inst.const_val is None:
                raise ValueError("CONST instruction missing const_val")
            self._const_table[inst.dest] = inst.const_val
            return []

        if op == "LOAD":
            if len(inst.srcs) != 1:
                raise ValueError("LOAD expects one source")
            src = inst.srcs[0]
            offset = self._input_offset(src, input_name_to_idx)
            self._live_regs.add(inst.dest)
            return [f"MOVSD({inst.dest}, [reg_in + {offset}])"]

        if op in {"ADD", "MUL", "SUB", "DIV", "POW"}:
            self._live_regs.add(inst.dest)
            return self._emit_binary(op, inst.dest, inst.srcs, input_name_to_idx)

        if op == "STORE":
            if len(inst.srcs) != 1:
                raise ValueError("STORE expects one source")
            out_idx = int(inst.dest.split("_")[1])
            src = self._value_ref(inst.srcs[0], input_name_to_idx)
            return [f"MOVSD([reg_out + {out_idx * 8}], {src})"]

        if len(inst.srcs) == 1:
            src = self._value_ref(inst.srcs[0], input_name_to_idx)
            self._live_regs.add(inst.dest)
            return [f"MOVSD({inst.dest}, {src})", f"# Unsupported unary op: {op}"]

        return [f"# Unsupported op: {op}, srcs={inst.srcs}"]

    def _emit_core(
        self, instructions: List[Instruction], input_name_to_idx: Dict[str, int]
    ) -> str:
        self._const_table = {}
        self._live_regs = set()
        lines: List[str] = []
        for inst in instructions:
            lines.extend(self._translate_instruction(inst, input_name_to_idx))
            for reg in inst.dead_vars:
                if self._is_reg(reg):
                    self._live_regs.discard(reg)
                    lines.append(f"# free {reg}")
        return "\n".join(lines)

    def render_to_string(
        self,
        model: MathModel,
        allocated_instructions: List[Instruction],
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(self.template_path.name)
        input_names = [str(s) for s in model.inputs]
        input_name_to_idx = {name: i for i, name in enumerate(input_names)}
        core = self._emit_core(allocated_instructions, input_name_to_idx)
        rendered = template.render(
            kernel_name=model.name,
            input_names=input_names,
            core_instructions=core,
        )
        return rendered

    def render(
        self,
        model: MathModel,
        allocated_instructions: List[Instruction],
        output_path: Path,
    ) -> Path:
        rendered = self.render_to_string(model, allocated_instructions)
        output_path = Path(output_path)
        output_path.write_text(rendered, encoding="utf-8")
        return output_path

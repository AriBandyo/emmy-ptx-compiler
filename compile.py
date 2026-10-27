import json
import sys
from pathlib import Path
from emmy.compiler.graph import Graph


ir_path = Path(sys.argv[1])


with ir_path.open() as f:
    graph = Graph.from_dict(json.load(f))


with (ir_path.parent / "meta.json").open() as f:
    meta = json.load(f)


kernel_meta = meta["kernels"][0]

kernel = graph.nodes[kernel_meta["node"]]
tile = kernel.op.body[0]
axis = tile.axes[0]
value_regs = {}
arithmetic_instructions = []
ptx_float_opcodes = {
    "add": "add.f32",
    "multiply": "mul.f32",
    "sub": "sub.f32"
}
instructions = []

def compiler_expr(expr):
    kind = type(expr).__name__

    
    if kind == "Var":
        return f"%r_{expr.name}"
        
    if kind == "Literal":

        return str(expr.value)


    if kind == "BinaryExpr":

        left = compiler_expr(expr.left)

        right = compiler_expr(expr.right)

        dest = f"%r{len(instructions) + 1}"

        opcode = {
            "*": "mul.lo.u32",
            "+": "add.u32"
        }[expr.op]

        instructions.append(
            f"{opcode} {dest}, {left}, {right} ;"
        )

        return dest

    raise TypeError(f"Unsupported expression: {kind}")
arg_order = kernel_meta["arg_order"]

pointer_instructions = []
body_instructions = []
shared_declarations = []

for j, arg_name in enumerate(arg_order):
    pointer_instructions.append(
        f"ld.param.u64 %rd_param{j}, [param_{arg_name}];"
    )

    pointer_instructions.append(
        f"cvta.to.global.u64 %rd_base{j}, %rd_param{j};"
    )
for i, op in enumerate(tile.body):

    

    arg_order = kernel_meta["arg_order"]
    if type(op).__name__ == "StridedLoop":

        # Initialize reduction accumulators
        for inner in op.body:
            if type(inner).__name__ == "Accum":
                if inner.op.name != "add":
                    raise NotImplementedError(
                        f"Unsupported accumulator: {inner.op.name}"
                    )

                acc_reg = f"%f{len(value_regs)}"
                value_regs[inner.name] = acc_reg

                body_instructions.append(
                    f"mov.f32 {acc_reg}, 0f00000000;"
                )

        start_reg = f"%r_{op.start.name}"
        loop_reg = f"%r_{op.axis.name}"
        loop_limit = op.axis.extent.as_static()
        loop_step = op.step.value
        body_instructions.extend([
            f"mov.u32 {loop_reg}, {start_reg};",
            f"LOOP_{i}:",
            f"setp.ge.u32 %p_loop, {loop_reg}, {loop_limit};",
            f"@%p_loop bra LOOP_END_{i};"
        ])
        for inner in op.body:
            if type(inner).__name__ == "Load":
                dest_reg = f"%f{len(value_regs)}"
                value_regs[inner.names[0]] = dest_reg

                param_index = arg_order.index(inner.input)
                start = len(instructions)

                if len(inner.index) == 2:
                    row = compiler_expr(inner.index[0])
                    col = compiler_expr(inner.index[1])

                    shape = next(
                        arg["shape"]
                        for arg in kernel_meta["args"]
                        if arg["name"] == inner.input
                    )

                    stride = shape[1]
                    index_reg = f"%r{len(instructions) + 1}"

                    instructions.append(
                        f"mad.lo.u32 {index_reg}, {row}, {stride}, {col};"
                    )
                else:
                    index_reg = compiler_expr(inner.index[0])

                body_instructions.extend(instructions[start:])

                body_instructions.extend([
                    f"mul.wide.u32 %rd_offset{i}, {index_reg}, {inner.dtype.nbytes};",
                    f"add.u64 %rd_addr{i}, %rd_base{param_index}, %rd_offset{i};",
                    f"ld.global.f32 {dest_reg}, [%rd_addr{i}];"
                ])

            elif type(inner).__name__ == "Assign":
                op_name = inner.op.name

                if op_name not in ptx_float_opcodes:
                    raise NotImplementedError(
                        f"Unsupported loop operation: {op_name}"
                    )

                dest_reg = f"%f{len(value_regs)}"
                value_regs[inner.name] = dest_reg

                left_reg = value_regs[inner.args[0]]
                right_reg = value_regs[inner.args[1]]

                opcode = ptx_float_opcodes[op_name]

                body_instructions.append(
                    f"{opcode} {dest_reg}, {left_reg}, {right_reg};"
                )
            elif type(inner).__name__ == "Accum":
                acc_reg = value_regs[inner.name]
                src_reg = value_regs[inner.value]

                instruction = (
                    f"add.f32 {acc_reg}, {acc_reg}, {src_reg};"
                )

                body_instructions.append(instruction)

            elif type(inner).__name__ == "Write":
                if len(inner.values) != 1:
                    raise NotImplementedError(
                        "Only scalar loop writes supported"
                    )

                source_reg = value_regs[inner.values[0]]
                output_index = arg_order.index(inner.output)

                start = len(instructions)

                if len(inner.index) == 2:
                    row = compiler_expr(inner.index[0])
                    col = compiler_expr(inner.index[1])

                    shape = next(
                        arg["shape"]
                        for arg in kernel_meta["args"]
                        if arg["name"] == inner.output
                    )

                    stride = shape[1]
                    index_reg = f"%r{len(instructions) + 1}"

                    instructions.append(
                        f"mad.lo.u32 {index_reg}, "
                        f"{row}, {stride}, {col};"
                    )

                elif len(inner.index) == 1:
                    index_reg = compiler_expr(inner.index[0])

                else:
                    raise NotImplementedError(
                        "Unsupported loop write index"
                    )

                body_instructions.extend(instructions[start:])

                body_instructions.extend([
                    f"mul.wide.u32 %rd_offset{i}, "
                    f"{index_reg}, {inner.value_dtype.nbytes};",

                    f"add.u64 %rd_addr{i}, "
                    f"%rd_base{output_index}, %rd_offset{i};",

                    f"st.global.f32 [%rd_addr{i}], {source_reg};"
                ])


        # Move to the next element
        body_instructions.extend([
            f"add.u32 {loop_reg}, {loop_reg}, {loop_step};",
            f"bra LOOP_{i};",
            f"LOOP_END_{i}:"
        ])

    elif type(op).__name__ == "Load":
        print("Load found at:", i, file=sys.stderr)
        print("Variable:",op.names[0], file=sys.stderr)
        print("Source",op.input, file=sys.stderr)
        print("Index", op.index[0], file=sys.stderr)
        value_regs[op.names[0]] = f"%f{len(value_regs)}"
        param_index = arg_order.index(op.input)
        start = len(instructions)
        index_reg = compiler_expr(op.index[0])

        body_instructions.extend(instructions[start:])

        print("Index operand:", index_reg, file=sys.stderr)
        print(
            "Parameter:", param_index,
            "Register:", value_regs[op.names[0]],
            file=sys.stderr
        )
        offset_reg = f"%rd_offset{i}"
        addr_reg = f"%rd_addr{i}"
        dest_reg = value_regs[op.names[0]]
        if index_reg == "0":
            body_instructions.append(
                f"mov.u64 {offset_reg}, 0;"
            )
        else:
            body_instructions.append(
                f"mul.wide.u32 {offset_reg}, {index_reg}, {op.dtype.nbytes};"
            )

        

        body_instructions.append(
            f"add.u64 {addr_reg}, %rd_base{param_index}, {offset_reg};"
        )

        body_instructions.append(
            f"ld.global.f32 {dest_reg}, [{addr_reg}];"
        )
        
    elif type(op).__name__ == "Assign":
        print("Assign found at:", i, file=sys.stderr)
        print("Result:", op.name, file=sys.stderr)
        print("Operation:", op.op.name, file=sys.stderr)
        print("Arguments:", op.args, file=sys.stderr)
        value_regs[op.name] = f"%f{len(value_regs)}"

        #condition for ReLU or else  op.args[1] when ReLU is applied will faile 
        dest_reg = value_regs[op.name]
        if op.op.name == "relu":
            src_reg = value_regs[op.args[0]]
            instruction = (
                f"max.f32 {dest_reg}, {src_reg},0f00000000;"
            )
        elif op.op.name in ("reciprocal", "sqrt"):
            src_reg = value_regs[op.args[0]]

            unary_opcodes = {
                "reciprocal": "rcp.rn.f32",
                "sqrt": "sqrt.rn.f32"
            }

            opcode = unary_opcodes[op.op.name]

            instruction = (
                f"{opcode} {dest_reg}, {src_reg};"
            )
        else:

            left_reg = value_regs[op.args[0]]
            right_reg = value_regs[op.args[1]]
            ptx_opcode = ptx_float_opcodes[op.op.name]
            instruction = (
                f"{ptx_opcode} {dest_reg}, "
                f"{left_reg}, {right_reg};"
            )
        print("Generated PTX:", instruction, file=sys.stderr)
        arithmetic_instructions.append(instruction)
        body_instructions.append(instruction)
    elif type(op).__name__ == "Write":
        print("Write found at:", i, file=sys.stderr)
        print("Output:", op.output, file=sys.stderr)
        print("Index:", op.index[0], file=sys.stderr)
        print("Values:", op.values, file=sys.stderr)
        output_regs = [value_regs[name] for name in op.values]

        output_param = arg_order.index(op.output)
        start = len(instructions)
        index_reg = compiler_expr(op.index[0])

        body_instructions.extend(instructions[start:])

        offset_reg = f"%rd_offset{i}"
        addr_reg = f"%rd_addr{i}"
        body_instructions.append(
            f"mul.wide.u32 {offset_reg}, {index_reg}, {op.value_dtype.nbytes};"
        )

        body_instructions.append(
            f"add.u64 {addr_reg}, %rd_base{output_param}, {offset_reg};"
        )
        regs = ", ".join(output_regs)

        width = len(output_regs)

        if width not in (2, 4):
            raise NotImplementedError(f"Unsupported vector width: {width}")

        body_instructions.append(
            f"st.global.v{width}.f32 [{addr_reg}], {{{regs}}};"
        )

        print("Output registers:", output_regs, file=sys.stderr)
        print("Output parameter:", output_param, file=sys.stderr)

    
    elif type(op).__name__ == "WarpShuffle":
        if len(op.state) != 1 or len(op.state_b) != 1:
            raise NotImplementedError(
                "Multiple shuffle states not supported yet"
            )

        combine = op.combine_states[0]
        if combine.op.name != "add":
            raise NotImplementedError(
                f"Unsupported shuffle operation: {combine.op.name}"
            )

        acc_reg = value_regs[op.state[0]]

        partner_reg = f"%f{len(value_regs)}"
        value_regs[op.state_b[0]] = partner_reg

        offset = op.length // 2

        while offset > 0:
            body_instructions.extend([
                f"mov.b32 %r_shfl_src, {acc_reg};",
                f"shfl.sync.bfly.b32 %r_shfl_dst, "
                f"%r_shfl_src, {offset}, 0x1f, 0xffffffff;",
                f"mov.b32 {partner_reg}, %r_shfl_dst;",
                f"add.f32 {acc_reg}, {acc_reg}, {partner_reg};"
            ])

            offset //= 2
    
    elif type(op).__name__ == "Smem":
        if op.dtype != "float":
            raise NotImplementedError(
                f"Unsupported shared dtype: {op.dtype}"
            )

        num_elements = 1
        for extent in op.extents:
            num_elements *= int(extent)

        size_bytes = num_elements * 4

        shared_declarations.append(
            f".shared .align 4 .b8 {op.name}[{size_bytes}];"
        )
    
    elif type(op).__name__ == "Cond":
        cond = op.cond

        if (
            cond.op == "=="
            and getattr(cond.left, "name", None) == "lane"
            and getattr(cond.right, "value", None) == 0
        ):
            write_op = op.body[0]

            if type(write_op).__name__ != "Write":
                raise NotImplementedError(
                    "Expected shared-memory Write"
                )

            source_reg = value_regs[write_op.values[0]]
            shared_name = write_op.output

            body_instructions.extend([
                "setp.ne.u32 %p_not_lane0, %r_lane, 0;",
                f"@%p_not_lane0 bra COND_END_{i};",
                f"mov.u32 %r_smem_base, {shared_name};",
                "mul.lo.u32 %r_smem_off, %r_warp, 4;",
                "add.u32 %r_smem_addr, %r_smem_base, %r_smem_off;",
                f"st.shared.f32 [%r_smem_addr], {source_reg};",
                f"COND_END_{i}:"
            ])

        
        elif (
            cond.op == "=="
            and getattr(cond.left, "name", None) in {a.name for a in tile.axes}
            and getattr(cond.right, "value", None) == 0
        ):
            
            # Check whether this thread should execute the output operations
            body_instructions.extend([
                f"setp.ne.u32 %p_not_output, "
                f"%r_{cond.left.name}, {cond.right.value};",
                f"@%p_not_output bra COND_END_{i};"
            ])

            # Process every operation inside this Cond
            for inner in op.body:

                if type(inner).__name__ == "Load":
                    if (
                        len(inner.index) != 1
                        or getattr(inner.index[0], "value", None) != 0
                    ):
                        raise NotImplementedError(
                            "Only scalar index-zero loads supported here"
                        )

                    dest_reg = f"%f{len(value_regs)}"
                    value_regs[inner.names[0]] = dest_reg

                    param_index = arg_order.index(inner.input)

                    body_instructions.append(
                        f"ld.global.f32 {dest_reg}, "
                        f"[%rd_base{param_index}];"
                    )

                elif type(inner).__name__ == "Assign":
                    if inner.op.name != "divide":
                        raise NotImplementedError(
                            f"Unsupported operation: {inner.op.name}"
                        )

                    dest_reg = f"%f{len(value_regs)}"
                    value_regs[inner.name] = dest_reg

                    left_reg = value_regs[inner.args[0]]
                    right_reg = value_regs[inner.args[1]]

                    body_instructions.append(
                        f"div.rn.f32 {dest_reg}, "
                        f"{left_reg}, {right_reg};"
                    )

                elif type(inner).__name__ == "Write":
                    source_reg = value_regs[inner.values[0]]
                    output_index = arg_order.index(inner.output)

                    if len(inner.index) == 0:
                        # Original scalar output
                        body_instructions.append(
                            f"st.global.f32 "
                            f"[%rd_base{output_index}], {source_reg};"
                        )

                    elif len(inner.index) == 1:
                        # Indexed output, e.g. sum_1[a0]
                        start = len(instructions)
                        index_reg = compiler_expr(inner.index[0])

                        body_instructions.extend(instructions[start:])

                        offset_reg = f"%rd_offset{i}"
                        addr_reg = f"%rd_addr{i}"

                        body_instructions.extend([
                            f"mul.wide.u32 {offset_reg}, "
                            f"{index_reg}, {inner.value_dtype.nbytes};",

                            f"add.u64 {addr_reg}, "
                            f"%rd_base{output_index}, {offset_reg};",

                            f"st.global.f32 [{addr_reg}], {source_reg};"
                        ])

                    else:
                        raise NotImplementedError(
                            "Multidimensional conditional writes not supported"
                        )

            body_instructions.append(f"COND_END_{i}:")


        else:
            raise NotImplementedError(
                f"Unsupported condition: {cond}"
            )


    
    elif type(op).__name__ == "TreeHalve":
        shared_name = op.bufs[0]
        acc_reg = value_regs[op.state[0]]
        temp_reg = value_regs[op.state_b[0]]

        # Only thread 0 combines the warp sums
        body_instructions.extend([
            "setp.ne.u32 %p_not_tree0, %r_tid, 0;",
            f"@%p_not_tree0 bra TREE_END_{i};",
            f"mov.u32 %r_smem_base, {shared_name};"
        ])

        # acc_reg already contains warp 0's sum
        for k in range(1, op.length):
            offset = k * 4

            body_instructions.extend([
                f"ld.shared.f32 {temp_reg}, "
                f"[%r_smem_base+{offset}];",
                f"add.f32 {acc_reg}, {acc_reg}, {temp_reg};"
            ])

       
        needs_broadcast = any(
            type(later).__name__ == "Assign"
            and op.state[0] in later.args
            for later in tile.body[i + 1:]
        )

        if needs_broadcast:
            body_instructions.extend([
                # Only thread 0 reaches this instruction
                f"st.shared.f32 [%r_smem_base], {acc_reg};",

                # All threads continue here
                f"TREE_END_{i}:",
                f"mov.u32 %r_smem_base, {shared_name};",
                "bar.sync 0;",
                f"ld.shared.f32 {acc_reg}, [%r_smem_base];"
            ])
        else:
            body_instructions.append(f"TREE_END_{i}:")



    elif type(op).__name__ == "Sync":
        body_instructions.append("bar.sync 0;")



    
print("Register mapping:", value_regs, file=sys.stderr)
print("Arithmetic PTX:", file=sys.stderr)
for inst in arithmetic_instructions:
    print(inst, file=sys.stderr)
print("Index PTX instructions:", instructions, file=sys.stderr)
print("Pointer PTX:", file=sys.stderr)
for inst in pointer_instructions:
    print(inst, file=sys.stderr)
print("Memory PTX:", file=sys.stderr)

for inst in body_instructions[:8]:
    print(inst, file=sys.stderr)
print("Ordered PTX:", file=sys.stderr)

for inst in body_instructions[:13]:
    print(inst, file=sys.stderr)
print("Final PTX instructions:", file=sys.stderr)

for inst in body_instructions[-5:]:
    print(inst, file=sys.stderr)
ptx_lines = [
    ".version 8.0",
    ".target sm_89",
    ".address_size 64",
    "",
    f".visible .entry {kernel_meta['entry']}("
]

for j, arg_name in enumerate(arg_order):
    comma = "," if j < len(arg_order) - 1 else ""

    ptx_lines.append(
        f"    .param .u64 param_{arg_name}{comma}"
    )

ptx_lines.append(")")
ptx_lines = [
    ".version 8.0",
    ".target sm_89",
    ".address_size 64",
    "",
    f".visible .entry {kernel_meta['entry']}("
]

for j, arg_name in enumerate(arg_order):
    comma = "," if j < len(arg_order) - 1 else ""
    ptx_lines.append(
        f"    .param .u64 param_{arg_name}{comma}"
    )

ptx_lines.append(")")


ptx_lines.append("{")
for declaration in shared_declarations:
    ptx_lines.append(f"    {declaration}")

ptx_lines.append(
    "    .reg .u32 %r_tid, %r_bid, %r_bdim, %r_a0;"
)
ptx_lines.append("    .reg .pred %p_out;")

# Registers for warp and lane identification
ptx_lines.append(
    "    .reg .u32 %r_lane, %r_warp;"
)

# Registers for shared-memory addressing
ptx_lines.append(
    "    .reg .u32 %r_smem_base, %r_smem_off, %r_smem_addr;"
)


ptx_lines.append(
    "    .reg .pred %p_not_lane0;"
)
ptx_lines.append("    .reg .pred %p_not_tree0;")
ptx_lines.append("    .reg .pred %p_not_output;")

ptx_lines.append("    .reg .pred %p_loop;")
if any(type(op).__name__ == "WarpShuffle" for op in tile.body):
    ptx_lines.append(
        "    .reg .b32 %r_shfl_src, %r_shfl_dst;"
    )

axis_names = {a.name for a in tile.axes}
axis_names.update(
    op.axis.name
    for op in tile.body
    if type(op).__name__ == "StridedLoop"
)

for name in sorted(axis_names):
    if name != "a0":
        ptx_lines.append(f"    .reg .u32 %r_{name};")

ptx_lines.append(f"    .reg .u32 %r<{len(instructions) + 1}>;")
ptx_lines.append(f"    .reg .u64 %rd_param<{len(arg_order)}>;")
ptx_lines.append(f"    .reg .u64 %rd_base<{len(arg_order)}>;")
ptx_lines.append(f"    .reg .u64 %rd_offset<{len(tile.body)}>;")
ptx_lines.append(f"    .reg .u64 %rd_addr<{len(tile.body)}>;")
ptx_lines.append(f"    .reg .f32 %f<{len(value_regs)}>;")
ptx_lines.extend([
    "    mov.u32 %r_tid, %tid.x;",
    "    mov.u32 %r_bid, %ctaid.x;",
    "    mov.u32 %r_bdim, %ntid.x;"
])

if len(tile.axes) == 2:
    # r04: one block processes one matrix row
    ptx_lines.extend([
        "    mov.u32 %r_a0, %r_bid;",
        f"    mov.u32 %r_{tile.axes[1].name}, %r_tid;",
        f"    setp.ge.u32 %p_out, %r_a0, {axis.extent.as_static()};",
        "    @%p_out bra DONE;"
    ])
else:
    # Preserve the mapping for our previous 1D problems
    ptx_lines.extend([
        "    mad.lo.u32 %r_a0, %r_bid, %r_bdim, %r_tid;",
        f"    setp.ge.u32 %p_out, %r_a0, {axis.extent.as_static()};",
        "    @%p_out bra DONE;"
    ])

if axis.name != "a0":
    ptx_lines.append(
        f"    mov.u32 %r_{axis.name}, %r_a0;"
    )

ptx_lines.extend([
    "    and.b32 %r_lane, %r_tid, 31;",
    "    shr.u32 %r_warp, %r_tid, 5;"
])

# Add kernel parameter pointers
for inst in pointer_instructions:
    ptx_lines.append(f"    {inst}")

# Add memory loads, arithmetic and output stores
for inst in body_instructions:
    ptx_lines.append(f"    {inst}")

# Finish the kernel
ptx_lines.extend([
    "DONE:",
    "    ret;",
    "}"
])

# Print complete PTX to standard output
for line in ptx_lines:
    print(line)

sys.exit(0)


#hardcoded 
first_load = tile.body[0]
second_load = tile.body[1]
first_assign = tile.body[2]
third_load = tile.body[3]
fourth_load = tile.body[4]
second_assign = tile.body[5]
write_op = tile.body[6]

print("Second load source:", second_load.input, file=sys.stderr)
print("Second load index:", second_load.index[0], file=sys.stderr)
print("Second load dtype:", second_load.dtype.name, file=sys.stderr)

print("Write type:", type(write_op).__name__, file=sys.stderr)
print("Write output:", write_op.output, file=sys.stderr)
print("Write index:", write_op.index[0], file=sys.stderr)
print("Write values:", write_op.values, file=sys.stderr)
print("Write dtype:", write_op.value_dtype.name, file=sys.stderr)

print("Assign type:", type(first_assign).__name__, file=sys.stderr)
print("Assign name:", first_assign.name, file=sys.stderr)
print("Assign operation:", first_assign.op, file=sys.stderr)
print("Assign arguments:", first_assign.args, file=sys.stderr)

print(
    "Operation attributes:",
    [name for name in dir(first_assign.op) if not name.startswith("_")],
    file=sys.stderr
)

float_opcodes = {
    "add": "add.f32",
    "mul": "mul.f32",
    "sub": "sub.f32"
}


operation_name = first_assign.op.name
opcode = float_opcodes[operation_name]

left_reg = value_regs[first_assign.args[0]]
right_reg = value_regs[first_assign.args[1]]
value_regs[first_assign.name] = "%f_result"
value_regs[third_load.names[0]] = "%f_in2"
value_regs[fourth_load.names[0]] = "%f_in3"
value_regs[second_assign.name] = "%f_result2"

left_reg2 = value_regs[second_assign.args[0]]
right_reg2 = value_regs[second_assign.args[1]]

opcode2 = float_opcodes[second_assign.op.name]



print("Third load:", third_load.input, third_load.index[0], file=sys.stderr)
print("Fourth load:", fourth_load.input, fourth_load.index[0], file=sys.stderr)

print("Second assign name:", second_assign.name, file=sys.stderr)
print("Second assign operation:", second_assign.op.name, file=sys.stderr)
print("Second assign arguments:", second_assign.args, file=sys.stderr)


index_register = compiler_expr(first_load.index[0])
second_index_register = compiler_expr(third_load.index[0])

print("Second index register:", second_index_register, file=sys.stderr)
print("All index instructions:", instructions, file=sys.stderr)

print("Generated index PTX:", instructions, file=sys.stderr)
print("Final index register:", index_register, file=sys.stderr)

print("Load source:", first_load.input, file=sys.stderr)
print("Index expression:", first_load.index[0], file=sys.stderr)
print("Element size:", first_load.dtype.nbytes, file=sys.stderr)

print("Tile axis:", axis.name, file=sys.stderr)
print("Tile extent:", axis.extent, file=sys.stderr)

print("Graph nodes:", list(graph.nodes), file=sys.stderr)
print("Kernel entry:", kernel_meta["entry"], file=sys.stderr)
print("Arguments:", kernel_meta["arg_order"], file=sys.stderr)
print("Grid:", kernel_meta["grid"], file=sys.stderr)
print("Block:", kernel_meta["block"], file=sys.stderr)


entry = kernel_meta["entry"]
args = kernel_meta["arg_order"]

print(".version 8.0")
print(".target sm_89")
print(".address_size 64")
print()

print(f".visible .entry {entry}(")

for i, name in enumerate(args):
    comma = "," if i < len(args) - 1 else ""
    print(f"    .param .u64 param_{name}{comma}")

print(")")
print("{")


print("    .reg .u32 %r_tid, %r_bid, %r_bdim, %r_a0;")
print("    .reg .pred %p_out;")
print(f"    .reg .u32 %r<{len(instructions) + 1}>;")
print(f"    .reg .u64 %rd<{len(args)}>;")
print("    .reg .u64 %rd_offset, %rd_addr0, %rd_gx0;")
print("    .reg .u64 %rd_addr1, %rd_gx1;")
print("    .reg .u64 %rd_offset2, %rd_addr2;")
print("    .reg .u64 %rd_addr3;")
print("    .reg .u64 %rd_gout, %rd_out_addr;")
print("    .reg .f32 %f_in0;")
print("    .reg .f32 %f_in1;")
print("    .reg .f32 %f_result;")
print("    .reg .f32 %f_in2;")
print("    .reg .f32 %f_in3;")
print("    .reg .f32 %f_result2;")
print("    mov.u32 %r_tid, %tid.x;")
print("    mov.u32 %r_bid, %ctaid.x;")
print("    mov.u32 %r_bdim, %ntid.x;")

print("    mad.lo.u32 %r_a0, %r_bid, %r_bdim, %r_tid;")
print(f"    setp.ge.u32 %p_out, %r_a0, {axis.extent};")
print("    @%p_out bra DONE;")
for i, name in enumerate(args):
    print(f"    ld.param.u64 %rd{i}, [param_{name}];")
print("    cvta.to.global.u64 %rd_gx0, %rd0;")
for instruction in instructions:
    print(instruction)
print(f"    mul.wide.u32 %rd_offset, {index_register}, {first_load.dtype.nbytes};")
print("    add.u64 %rd_addr0, %rd_gx0, %rd_offset;")
print("    ld.global.f32 %f_in0, [%rd_addr0];")
source_index = args.index(second_load.input)

print(f"    cvta.to.global.u64 %rd_gx1, %rd{source_index};")
print("    add.u64 %rd_addr1, %rd_gx1, %rd_offset;")
print("    ld.global.f32 %f_in1, [%rd_addr1];")
print(f"    {opcode} {value_regs[first_assign.name]}, {left_reg}, {right_reg};")
print(f"    mul.wide.u32 %rd_offset2, {second_index_register}, {third_load.dtype.nbytes};")
print("    add.u64 %rd_addr2, %rd_gx0, %rd_offset2;")
print("    ld.global.f32 %f_in2, [%rd_addr2];")
print("    add.u64 %rd_addr3, %rd_gx1, %rd_offset2;")
print("    ld.global.f32 %f_in3, [%rd_addr3];")
print(
    f"    {opcode2} {value_regs[second_assign.name]}, "
    f"{left_reg2}, {right_reg2};"
)

# The current Write uses the same starting index as our first Load.
if write_op.index[0] != first_load.index[0]:
    raise NotImplementedError("Different output index not implemented yet")

output_index = args.index(write_op.output)

out0 = value_regs[write_op.values[0]]
out1 = value_regs[write_op.values[1]]

print(f"    cvta.to.global.u64 %rd_gout, %rd{output_index};")
print("    add.u64 %rd_out_addr, %rd_gout, %rd_offset;")
print(f"    st.global.v2.f32 [%rd_out_addr], {{{out0}, {out1}}};")

print("DONE:")
print("    ret;")
print("}")
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
first_load = tile.body[0]

second_load = tile.body[1]

print("Second load source:", second_load.input, file=sys.stderr)
print("Second load index:", second_load.index[0], file=sys.stderr)
print("Second load dtype:", second_load.dtype.name, file=sys.stderr)

first_assign = tile.body[2]
third_load = tile.body[3]
fourth_load = tile.body[4]
second_assign = tile.body[5]
write_op = tile.body[6]

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
value_regs = {
    first_load.names[0]: "%f_in0",
    second_load.names[0]: "%f_in1"
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
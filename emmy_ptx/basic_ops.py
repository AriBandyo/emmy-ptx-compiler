import sys
from emmy_ptx.elementwise import emit_assign


def emit_basic_op(
    op, i, *,
    value_regs, instructions, body_instructions,
    compiler_expr, kernel_meta, arg_order,
    ptx_float_opcodes, arithmetic_instructions,
):
    if type(op).__name__ == "Load":
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
        generated = emit_assign(op, value_regs)
        arithmetic_instructions.extend(generated)
        body_instructions.extend(generated)

    elif type(op).__name__ == "Write":
        print("Write found at:", i, file=sys.stderr)
        print("Output:", op.output, file=sys.stderr)
        print("Index:", op.index[0], file=sys.stderr)
        print("Values:", op.values, file=sys.stderr)
        output_regs = [value_regs[name] for name in op.values]

        output_param = arg_order.index(op.output)
        start = len(instructions)

        if len(op.index) == 2:
            row_reg = compiler_expr(op.index[0])
            col_reg = compiler_expr(op.index[1])

            output_shape = next(
                arg["shape"]
                for arg in kernel_meta["args"]
                if arg["name"] == op.output
            )

            index_reg = f"%r{len(instructions) + 1}"

            instructions.append(
                f"mad.lo.u32 {index_reg}, "
                f"{row_reg}, {output_shape[1]}, {col_reg};"
            )

        elif len(op.index) == 1:
            index_reg = compiler_expr(op.index[0])

        else:
            raise NotImplementedError(
                "Unsupported top-level write index"
            )

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

        if op.atomic:
            if width != 1 or op.value_dtype.name != "f32":
                raise NotImplementedError("Unsupported atomic write")

            body_instructions.append(
                f"red.global.add.f32 [{addr_reg}], {output_regs[0]};"
            )

        elif width == 1:
            body_instructions.append(
                f"st.global.f32 [{addr_reg}], {output_regs[0]};"
            )

        elif width in (2, 4):
            body_instructions.append(
                f"st.global.v{width}.f32 [{addr_reg}], {{{regs}}};"
            )

        else:
            raise NotImplementedError(f"Unsupported vector width: {width}")

        print("Output registers:", output_regs, file=sys.stderr)
        print("Output parameter:", output_param, file=sys.stderr)

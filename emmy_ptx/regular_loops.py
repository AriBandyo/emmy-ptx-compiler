import struct

def make_emit_regular_loop(body_instructions, value_regs, instructions, compiler_expr, kernel_meta, arg_order, ptx_float_opcodes):
    regular_loop_counter = 0

    def emit_regular_loop(node):
        nonlocal regular_loop_counter

        loop_id = regular_loop_counter
        regular_loop_counter += 1

        axis_reg = f"%r_{node.axis.name}"
        loop_label = f"REGULAR_LOOP_{loop_id}"
        loop_limit = node.axis.extent.as_static()

        for stmt in node.body:
            if type(stmt).__name__ != "Accum":
                continue

            if stmt.name not in value_regs:
                value_regs[stmt.name] = f"%f{len(value_regs)}"

            identities = {
                "add": "0f00000000",
                "maximum": "0fF149F2CA"
            }

            if stmt.op.name not in identities:
                raise NotImplementedError(
                    f"Unsupported loop accumulator: {stmt.op.name}"
                )

            body_instructions.append(
                f"mov.f32 {value_regs[stmt.name]}, "
                f"{identities[stmt.op.name]};"
            )

        body_instructions.extend([
            f"mov.u32 {axis_reg}, 0;",
            f"{loop_label}:",
            f"setp.ge.u32 %p_inner_loop, {axis_reg}, {loop_limit};",
            f"@%p_inner_loop bra {loop_label}_END;"
        ])

        for stmt in node.body:
            if type(stmt).__name__ == "Loop":
                emit_regular_loop(stmt)

            elif type(stmt).__name__ == "Load":

                if stmt.dtype.name != "f32":
                    raise NotImplementedError(
                        f"Unsupported loop load dtype: {stmt.dtype.name}"
                    )

                start = len(instructions)

                if len(stmt.index) == 2:
                    row_reg = compiler_expr(stmt.index[0])
                    col_reg = compiler_expr(stmt.index[1])

                    input_shape = next(
                        arg["shape"]
                        for arg in kernel_meta["args"]
                        if arg["name"] == stmt.input
                    )

                    index_reg = f"%r{len(instructions) + 1}"

                    instructions.append(
                        f"mad.lo.u32 {index_reg}, "
                        f"{row_reg}, {input_shape[1]}, {col_reg};"
                    )

                elif len(stmt.index) == 1:
                    index_reg = compiler_expr(stmt.index[0])

                else:
                    raise NotImplementedError(
                        "Unsupported regular loop load index"
                    )

                body_instructions.extend(instructions[start:])

                param_index = arg_order.index(stmt.input)

                body_instructions.extend([
                    f"mul.wide.u32 %rd_regular_offset, {index_reg}, 4;",
                    f"add.u64 %rd_regular_addr, "
                    f"%rd_base{param_index}, %rd_regular_offset;"
                ])

                load_regs = []

                for name in stmt.names:
                    dest_reg = f"%f{len(value_regs)}"
                    value_regs[name] = dest_reg
                    load_regs.append(dest_reg)

                if len(load_regs) == 1:
                    body_instructions.append(
                        f"ld.global.f32 {load_regs[0]}, "
                        f"[%rd_regular_addr];"
                    )

                elif len(load_regs) == 2:
                    regs = ", ".join(load_regs)
                    body_instructions.append(
                        f"ld.global.v2.f32 {{{regs}}}, "
                        f"[%rd_regular_addr];"
                    )

                else:
                    raise NotImplementedError(
                        f"Unsupported loop load width: {len(load_regs)}"
                    )


        
            elif type(stmt).__name__ == "Assign":

                op_name = stmt.op.name

                dest_reg = f"%f{len(value_regs)}"
                value_regs[stmt.name] = dest_reg

                if op_name == "exp":

                    src_reg = value_regs[stmt.args[0]]

                    body_instructions.extend([
                        f"mul.f32 {dest_reg}, {src_reg}, 0f3FB8AA3B;",
                        f"ex2.approx.f32 {dest_reg}, {dest_reg};"
                    ])

                elif op_name in ptx_float_opcodes:

                    left_reg = value_regs[stmt.args[0]]
                    right_reg = value_regs[stmt.args[1]]

                    opcode = ptx_float_opcodes[op_name]

                    body_instructions.append(
                        f"{opcode} {dest_reg}, {left_reg}, {right_reg};"
                    )

                else:
                    raise NotImplementedError(
                        f"Unsupported regular loop operation: {op_name}"
                    )
            elif type(stmt).__name__ == "Let":

                if type(stmt.value).__name__ != "Literal":
                    raise NotImplementedError(
                        "Unsupported regular loop Let"
                    )

                dest_reg = f"%f{len(value_regs)}"
                value_regs[stmt.name] = dest_reg

                bits = struct.unpack(
                    "<I",
                    struct.pack("<f", float(stmt.value.value))
                )[0]

                body_instructions.append(
                    f"mov.f32 {dest_reg}, 0f{bits:08X};"
                )

            elif type(stmt).__name__ == "Accum":

                acc_reg = value_regs[stmt.name]
                src_reg = value_regs[stmt.value]

                base_reg = (
                    value_regs[stmt.base]
                    if stmt.base is not None
                    else acc_reg
                )

                accumulator_opcodes = {
                    "add": "add.f32",
                    "maximum": "max.f32"
                }

                if stmt.op.name not in accumulator_opcodes:
                    raise NotImplementedError(
                        f"Unsupported regular loop accumulator: {stmt.op.name}"
                    )

                opcode = accumulator_opcodes[stmt.op.name]

                body_instructions.append(
                    f"{opcode} {acc_reg}, {base_reg}, {src_reg};"
                )

            else:
                raise NotImplementedError(
                    f"Unsupported regular loop statement: "
                    f"{type(stmt).__name__}"
                )

        body_instructions.extend([
            f"add.u32 {axis_reg}, {axis_reg}, 1;",
            f"bra {loop_label};",
            f"{loop_label}_END:"
        ])

    return emit_regular_loop

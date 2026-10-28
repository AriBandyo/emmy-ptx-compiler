from emmy_ptx.elementwise import emit_assign
import struct
def emit_strided_loop(
    op, i, *,
    value_regs, instructions, body_instructions,
    compiler_expr, kernel_meta, arg_order, ptx_float_opcodes,
    shared_shapes, shared_offsets, use_dynamic_shared, emit_cp_async,
):


    for inner in op.body:
        if type(inner).__name__ == "Accum":

            identities = {
                "add": "0f00000000",
                "maximum": "0fF149F2CA"
            }

            if inner.op.name not in identities:
                raise NotImplementedError(
                    f"Unsupported accumulator: {inner.op.name}"
                )

            acc_reg = f"%f{len(value_regs)}"
            value_regs[inner.name] = acc_reg

            identity = identities[inner.op.name]

            body_instructions.append(
                f"mov.f32 {acc_reg}, {identity};"
            )

    start_reg = compiler_expr(op.start)
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

        if type(inner).__name__ == "Let":

            if type(inner.value).__name__ != "Literal":
                raise NotImplementedError(
                    "Unsupported Let expression"
                )

            dest_reg = f"%f{len(value_regs)}"
            value_regs[inner.name] = dest_reg

            bits = struct.unpack(
                "<I",
                struct.pack("<f", float(inner.value.value))
            )[0]

            body_instructions.append(
                f"mov.f32 {dest_reg}, 0f{bits:08X};"
            )

        elif type(inner).__name__ == "Load":

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
                    f"mad.lo.u32 {index_reg}, "
                    f"{row}, {stride}, {col};"
                )

            elif len(inner.index) == 1:
                index_reg = compiler_expr(inner.index[0])

            else:
                raise NotImplementedError(
                    "Unsupported loop load index"
                )

            body_instructions.extend(instructions[start:])

            body_instructions.extend([
                f"mul.wide.u32 %rd_offset{i}, "
                f"{index_reg}, {inner.dtype.nbytes};",

                f"add.u64 %rd_addr{i}, "
                f"%rd_base{param_index}, %rd_offset{i};",

                f"ld.global.f32 {dest_reg}, [%rd_addr{i}];"
            ])

        elif type(inner).__name__ == "Assign":
            body_instructions.extend(
                emit_assign(inner, value_regs)
            )

        elif type(inner).__name__ == "Accum":

            acc_reg = value_regs[inner.name]
            src_reg = value_regs[inner.value]

            if inner.base is not None:
                base_reg = value_regs[inner.base]
            else:
                base_reg = acc_reg

            accumulator_opcodes = {
                "add": "add.f32",
                "maximum": "max.f32"
            }

            if inner.op.name not in accumulator_opcodes:
                raise NotImplementedError(
                    f"Unsupported accumulator: {inner.op.name}"
                )

            opcode = accumulator_opcodes[inner.op.name]

            body_instructions.append(
                f"{opcode} {acc_reg}, "
                f"{base_reg}, {src_reg};"
            )

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

        elif type(inner).__name__ == "StridedLoop":

            start = len(instructions)
            inner_start = compiler_expr(inner.start)
            body_instructions.extend(instructions[start:])

            inner_reg = f"%r_{inner.axis.name}"
            inner_limit = inner.axis.extent.as_static()
            inner_step = inner.step.value
            inner_label = f"ASYNC_{i}_{len(body_instructions)}"

            body_instructions.extend([
                f"mov.u32 {inner_reg}, {inner_start};",
                f"{inner_label}:",
                f"setp.ge.u32 %p_async_loop, {inner_reg}, {inner_limit};",
                f"@%p_async_loop bra {inner_label}_END;"
            ])

            for stmt in inner.body:
                if type(stmt).__name__ == "CpAsyncCopy":
                    emit_cp_async(stmt)
                else:
                    raise NotImplementedError(
                        f"Unsupported nested strided operation: "
                        f"{type(stmt).__name__}"
                    )

            body_instructions.extend([
                f"add.u32 {inner_reg}, {inner_reg}, {inner_step};",
                f"bra {inner_label};",
                f"{inner_label}_END:"
            ])
        elif type(inner).__name__ == "Loop":

            nested_reg = f"%r_{inner.axis.name}"
            nested_limit = inner.axis.extent.as_static()
            nested_label = f"INNER_{i}_{len(body_instructions)}"

            body_instructions.extend([
                f"mov.u32 {nested_reg}, 0;",
                f"{nested_label}:",
                f"setp.ge.u32 %p_inner_loop, {nested_reg}, {nested_limit};",
                f"@%p_inner_loop bra {nested_label}_END;"
            ])

            for stmt in inner.body:

                if type(stmt).__name__ == "Load":

                    if stmt.input not in shared_shapes:
                        raise NotImplementedError(
                            f"Unsupported nested load: {stmt.input}"
                        )

                    start = len(instructions)

                    row_reg = compiler_expr(stmt.index[0])
                    col_reg = compiler_expr(stmt.index[1])
                    stride = shared_shapes[stmt.input][1]

                    index_reg = f"%r{len(instructions) + 1}"

                    instructions.append(
                        f"mad.lo.u32 {index_reg}, "
                        f"{row_reg}, {stride}, {col_reg};"
                    )

                    body_instructions.extend(instructions[start:])

                    dest_reg = f"%f{len(value_regs)}"
                    value_regs[stmt.names[0]] = dest_reg

                    if use_dynamic_shared:
                        offset = shared_offsets[stmt.input]
                        shared_base = (
                            "_smem_pool"
                            if offset == 0
                            else f"_smem_pool+{offset}"
                        )
                    else:
                        shared_base = stmt.input

                    body_instructions.extend([
                        f"mov.u32 %r_smem_base, {shared_base};",
                        f"mul.lo.u32 %r_smem_off, {index_reg}, 4;",
                        f"add.u32 %r_smem_addr, "
                        f"%r_smem_base, %r_smem_off;",
                        f"ld.shared.f32 {dest_reg}, [%r_smem_addr];"
                    ])

                elif type(stmt).__name__ == "Assign":
                    body_instructions.extend(
                        emit_assign(stmt, value_regs)
                    )

                elif type(stmt).__name__ == "Accum":

                    acc_reg = value_regs[stmt.name]
                    src_reg = value_regs[stmt.value]

                    base_reg = (
                        value_regs[stmt.base]
                        if stmt.base is not None
                        else acc_reg
                    )

                    if stmt.op.name != "add":
                        raise NotImplementedError(
                            f"Unsupported nested accumulator: {stmt.op.name}"
                        )

                    body_instructions.append(
                        f"add.f32 {acc_reg}, {base_reg}, {src_reg};"
                    )

                else:
                    raise NotImplementedError(
                        f"Unsupported nested statement: "
                        f"{type(stmt).__name__}"
                    )

            body_instructions.extend([
                f"add.u32 {nested_reg}, {nested_reg}, 1;",
                f"bra {nested_label};",
                f"{nested_label}_END:"
            ])

        elif type(inner).__name__ == "Sync":
            body_instructions.append("bar.sync 0;")
        elif type(inner).__name__ == "CpAsyncCopy":
            emit_cp_async(inner)

        elif type(inner).__name__ == "CpAsyncCommit":
            body_instructions.append(
                "cp.async.commit_group;"
            )

        elif type(inner).__name__ == "CpAsyncWait":
            body_instructions.append(
                f"cp.async.wait_group {inner.group};"
            )


    # Move to the next element
    body_instructions.extend([
        f"add.u32 {loop_reg}, {loop_reg}, {loop_step};",
        f"bra LOOP_{i};",
        f"LOOP_END_{i}:"
    ])

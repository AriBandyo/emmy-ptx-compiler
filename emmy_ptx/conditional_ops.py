def emit_cond(
    op, i, *, tile, value_regs, instructions,
    body_instructions, compiler_expr, arg_order, kernel_meta
):
    cond = op.cond

    if (
        cond.op == "=="
        and getattr(cond.left, "name", None) == "lane"
        and getattr(cond.right, "value", None) == 0
    ):
        body_instructions.extend([
            "setp.ne.u32 %p_not_lane0, %r_lane, 0;",
            f"@%p_not_lane0 bra COND_END_{i};"
        ])

        for write_op in op.body:
            if type(write_op).__name__ != "Write":
                raise NotImplementedError(
                    "Expected shared-memory Write"
                )

            if len(write_op.values) != 1:
                raise NotImplementedError(
                    "Only scalar shared-memory writes supported"
                )

            if (
                len(write_op.index) != 1
                or getattr(write_op.index[0], "name", None) != "warp"
            ):
                raise NotImplementedError(
                    "Unsupported shared-memory write index"
                )

            source_reg = value_regs[write_op.values[0]]
            shared_name = write_op.output

            body_instructions.extend([
                f"mov.u32 %r_smem_base, {shared_name};",
                "mul.lo.u32 %r_smem_off, %r_warp, 4;",
                "add.u32 %r_smem_addr, %r_smem_base, %r_smem_off;",
                f"st.shared.f32 [%r_smem_addr], {source_reg};"
            ])

        body_instructions.append(f"COND_END_{i}:")

    elif (
        cond.op == "=="
        and getattr(cond.left, "name", None) in {a.name for a in tile.axes}
        and getattr(cond.right, "value", None) == 0
    ):
        body_instructions.extend([
            f"setp.ne.u32 %p_not_output, "
            f"%r_{cond.left.name}, {cond.right.value};",
            f"@%p_not_output bra COND_END_{i};"
        ])

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
                    body_instructions.append(
                        f"st.global.f32 "
                        f"[%rd_base{output_index}], {source_reg};"
                    )

                elif len(inner.index) == 1:
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

            else:
                raise NotImplementedError(
                    f"Unsupported conditional operation: "
                    f"{type(inner).__name__}"
                )

        body_instructions.append(f"COND_END_{i}:")

    else:
        raise NotImplementedError(
            f"Unsupported condition: {cond}"
        )

def compile_file(ir_filename):
    import json
    import sys
    import struct
    from pathlib import Path
    from emmy.compiler.graph import Graph
    from emmy_ptx.ptx import build_ptx
    from emmy_ptx.conditional_ops import emit_cond
    from emmy_ptx.memory_ops import emit_smem, emit_sync
    from emmy_ptx.basic_ops import emit_basic_op
    from emmy_ptx.loop_ops import emit_strided_loop


    ir_path = Path(ir_filename)


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
        "sub": "sub.f32",
        "subtract": "sub.f32",
        "maximum": "max.f32"
    }
    instructions = []

    from emmy_ptx.expressions import make_compiler_expr
    compiler_expr = make_compiler_expr(instructions)

    arg_order = kernel_meta["arg_order"]

    pointer_instructions = []
    body_instructions = []
    shared_declarations = []
    shared_shapes = {
        op.name: op.extents
        for op in tile.body
        if type(op).__name__ == "Smem"
    }
    shared_offsets = {}
    shared_bytes = 0
    pool_alignment = 16

    for sm in tile.body:
        if type(sm).__name__ == "Smem":
            alignment = max(4, int(sm.align or 4))
            pool_alignment = max(pool_alignment, alignment)

            shared_bytes = (
                (shared_bytes + alignment - 1) // alignment
            ) * alignment

            shared_offsets[sm.name] = shared_bytes

            elements = 1
            for extent in sm.extents:
                elements *= int(extent)

            shared_bytes += elements * 4

    use_dynamic_shared = shared_bytes > 49152
    from emmy_ptx.async_copy import make_emit_cp_async
    emit_cp_async = make_emit_cp_async(
        instructions, compiler_expr, kernel_meta, arg_order,
        shared_shapes, body_instructions, use_dynamic_shared,
        shared_offsets
    )

    for j, arg_name in enumerate(arg_order):
        pointer_instructions.append(
            f"ld.param.u64 %rd_param{j}, [param_{arg_name}];"
        )

        pointer_instructions.append(
            f"cvta.to.global.u64 %rd_base{j}, %rd_param{j};"
        )

    regular_loop_counter = 0

    from emmy_ptx.regular_loops import make_emit_regular_loop
    emit_regular_loop = make_emit_regular_loop(
        body_instructions, value_regs, instructions,
        compiler_expr, kernel_meta, arg_order,
        ptx_float_opcodes
    )

    for i, op in enumerate(tile.body):

    
        arg_order = kernel_meta["arg_order"]

        if type(op).__name__ == "Init":

            dest_reg = f"%f{len(value_regs)}"
            value_regs[op.name] = dest_reg

            bits = struct.unpack(
                "<I",
                struct.pack("<f", float(op.identity))
            )[0]

            body_instructions.append(
                f"mov.f32 {dest_reg}, 0f{bits:08X};"
            )
        elif type(op).__name__ == "Loop":
            emit_regular_loop(op)

        elif type(op).__name__ == "StridedLoop":
            emit_strided_loop(
                op, i,
                value_regs=value_regs,
                instructions=instructions,
                body_instructions=body_instructions,
                compiler_expr=compiler_expr,
                kernel_meta=kernel_meta,
                arg_order=arg_order,
                ptx_float_opcodes=ptx_float_opcodes,
                shared_shapes=shared_shapes,
                shared_offsets=shared_offsets,
                use_dynamic_shared=use_dynamic_shared,
                emit_cp_async=emit_cp_async,
            )

        elif type(op).__name__ in ("Load", "Assign", "Write"):
            emit_basic_op(
                op, i,
                value_regs=value_regs,
                instructions=instructions,
                body_instructions=body_instructions,
                compiler_expr=compiler_expr,
                kernel_meta=kernel_meta,
                arg_order=arg_order,
                ptx_float_opcodes=ptx_float_opcodes,
                arithmetic_instructions=arithmetic_instructions,
            )

        elif type(op).__name__ == "WarpShuffle":

            if len(op.state) != len(op.state_b):
                raise NotImplementedError(
                    "Mismatched shuffle states"
                )

            for partner_name in op.state_b:
                if partner_name not in value_regs:
                    value_regs[partner_name] = f"%f{len(value_regs)}"

            offset = op.length // 2

            while offset > 0:

                for state_name, partner_name in zip(
                    op.state, op.state_b
                ):
                    src_reg = value_regs[state_name]
                    partner_reg = value_regs[partner_name]

                    body_instructions.extend([
                        f"mov.b32 %r_shfl_src, {src_reg};",
                        f"shfl.sync.bfly.b32 %r_shfl_dst, "
                        f"%r_shfl_src, {offset}, 0x1f, 0xffffffff;",
                        f"mov.b32 {partner_reg}, %r_shfl_dst;"
                    ])

                for combine in op.combine_states:

                    op_name = combine.op.name

                    if combine.name not in value_regs:
                        value_regs[combine.name] = f"%f{len(value_regs)}"

                    dest_reg = value_regs[combine.name]

                    if op_name == "copy":

                        src_reg = value_regs[combine.args[0]]

                        body_instructions.append(
                            f"mov.f32 {dest_reg}, {src_reg};"
                        )

                    elif op_name == "exp":

                        src_reg = value_regs[combine.args[0]]

                        body_instructions.extend([
                            f"mul.f32 {dest_reg}, "
                            f"{src_reg}, 0f3FB8AA3B;",
                            f"ex2.approx.f32 {dest_reg}, {dest_reg};"
                        ])

                    else:

                        if op_name not in ptx_float_opcodes:
                            raise NotImplementedError(
                                f"Unsupported shuffle operation: {op_name}"
                            )

                        left_reg = value_regs[combine.args[0]]
                        right_reg = value_regs[combine.args[1]]

                        opcode = ptx_float_opcodes[op_name]

                        body_instructions.append(
                            f"{opcode} {dest_reg}, "
                            f"{left_reg}, {right_reg};"
                        )

                offset //= 2

        elif type(op).__name__ == "CpAsyncCopy":
            emit_cp_async(op)

        elif type(op).__name__ == "CpAsyncCommit":
            body_instructions.append(
                "cp.async.commit_group;"
            )

        elif type(op).__name__ == "CpAsyncWait":
            body_instructions.append(
                f"cp.async.wait_group {op.group};"
            )
        elif type(op).__name__ == "Smem":
            emit_smem(op, shared_declarations)

        elif type(op).__name__ == "Cond":
            emit_cond(
                op, i,
                tile=tile,
                value_regs=value_regs,
                instructions=instructions,
                body_instructions=body_instructions,
                compiler_expr=compiler_expr,
                arg_order=arg_order,
                kernel_meta=kernel_meta,
            )

        elif type(op).__name__ == "TreeHalve":

            if len(op.state) > 1:

                if not (
                    len(op.state) == len(op.state_b) == len(op.bufs)
                ):
                    raise NotImplementedError(
                        "Mismatched tree reduction states"
                    )

                for name in op.state_b:
                    if name not in value_regs:
                        value_regs[name] = f"%f{len(value_regs)}"

                body_instructions.extend([
                    "setp.ne.u32 %p_not_tree0, %r_tid, 0;",
                    f"@%p_not_tree0 bra TREE_END_{i};"
                ])

                for k in range(1, op.length):

                    for shared_name, partner_name in zip(
                        op.bufs, op.state_b
                    ):
                        partner_reg = value_regs[partner_name]

                        body_instructions.extend([
                            f"mov.u32 %r_smem_base, {shared_name};",
                            f"ld.shared.f32 {partner_reg}, "
                            f"[%r_smem_base+{k * 4}];"
                        ])

                    for combine in op.combine_states:

                        op_name = combine.op.name

                        if combine.name not in value_regs:
                            value_regs[combine.name] = f"%f{len(value_regs)}"

                        dest_reg = value_regs[combine.name]

                        if op_name == "copy":
                            src_reg = value_regs[combine.args[0]]

                            body_instructions.append(
                                f"mov.f32 {dest_reg}, {src_reg};"
                            )

                        elif op_name == "exp":
                            src_reg = value_regs[combine.args[0]]

                            body_instructions.extend([
                                f"mul.f32 {dest_reg}, "
                                f"{src_reg}, 0f3FB8AA3B;",
                                f"ex2.approx.f32 {dest_reg}, {dest_reg};"
                            ])

                        else:
                            if op_name not in ptx_float_opcodes:
                                raise NotImplementedError(
                                    f"Unsupported tree operation: {op_name}"
                                )

                            left_reg = value_regs[combine.args[0]]
                            right_reg = value_regs[combine.args[1]]
                            opcode = ptx_float_opcodes[op_name]

                            body_instructions.append(
                                f"{opcode} {dest_reg}, "
                                f"{left_reg}, {right_reg};"
                            )

                for state_name, shared_name in zip(
                    op.state, op.bufs
                ):
                    acc_reg = value_regs[state_name]

                    body_instructions.extend([
                        f"mov.u32 %r_smem_base, {shared_name};",
                        f"st.shared.f32 [%r_smem_base], {acc_reg};"
                    ])

                body_instructions.extend([
                    f"TREE_END_{i}:",
                    "bar.sync 0;"
                ])

                for state_name, shared_name in zip(
                    op.state, op.bufs
                ):
                    acc_reg = value_regs[state_name]

                    body_instructions.extend([
                        f"mov.u32 %r_smem_base, {shared_name};",
                        f"ld.shared.f32 {acc_reg}, [%r_smem_base];"
                    ])

            else:

                shared_name = op.bufs[0]
                acc_reg = value_regs[op.state[0]]
                temp_reg = value_regs[op.state_b[0]]

                body_instructions.extend([
                    "setp.ne.u32 %p_not_tree0, %r_tid, 0;",
                    f"@%p_not_tree0 bra TREE_END_{i};",
                    f"mov.u32 %r_smem_base, {shared_name};"
                ])

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
                        f"st.shared.f32 [%r_smem_base], {acc_reg};",
                        f"TREE_END_{i}:",
                        f"mov.u32 %r_smem_base, {shared_name};",
                        "bar.sync 0;",
                        f"ld.shared.f32 {acc_reg}, [%r_smem_base];"
                    ])
                else:
                    body_instructions.append(f"TREE_END_{i}:")


        elif type(op).__name__ == "Sync":
            emit_sync(body_instructions)



    
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
    sys.stdout.write(build_ptx(
        kernel_meta=kernel_meta,
        arg_order=arg_order,
        tile=tile,
        axis=axis,
        use_dynamic_shared=use_dynamic_shared,
        pool_alignment=pool_alignment,
        shared_declarations=shared_declarations,
        instructions=instructions,
        pointer_instructions=pointer_instructions,
        body_instructions=body_instructions,
        value_regs=value_regs,
    ))
